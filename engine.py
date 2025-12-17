import time
import uuid
import json
import re
import math
import requests
import os
from datetime import datetime, timedelta, timezone
from collections import Counter
from functools import lru_cache
from typing import Optional, Dict, Any, List, Tuple

# --- API IMPORTS ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supabase import create_client, Client
# 🔴 CHANGED: Removed Ollama, Added Groq
from langchain_groq import ChatGroq 
from sentence_transformers import SentenceTransformer
import config

# --- CONFIGURATION ---
SYSTEM_NAME = "Suraksha Sentinel v102 (Groq Cloud)"
# We force a Groq-supported model (Llama3 is best on Groq)
GROQ_MODEL = "llama3-70b-8192" 
SAFE_LIMIT = 20 
GEO_API_KEY = getattr(config, "GEO_API_KEY", None)
GROQ_API_KEY = getattr(config, "GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# =========================================================================
# 📚 DEFINITIONS
# =========================================================================
DEFINITIONS = {
    "active_sos": { "table": "incidents", "filters": [("type", "eq", "sos"), ("status", "neq", "resolved")] },
    "total_sos": { "table": "incidents", "filters": [("type", "eq", "sos")] },
    "danger_zones": { "table": "zones", "filters": [] },
    "solo_users": { "table": "users", "filters": [("group_id", "is", "null")] },
    "group_users": { "table": "users", "filters": [("group_id", "not.is", "null")] },
    "active_groups": { "table": "groups", "filters": [] }
}

class SentinelEngine:
    def __init__(self):
        print(f">> ⚙️ Initializing {SYSTEM_NAME}...")
        
        # 1. Database Connections
        self.db_live: Client = create_client(config.URL_A, config.KEY_A)
        self.db_brain: Client = create_client(config.URL_B, config.KEY_B)
        
        # 2. Initialize Groq LLM (Cloud) instead of Ollama (Local)
        if not GROQ_API_KEY:
            print(">> ⚠️ CRITICAL: GROQ_API_KEY not found in config or env!")
        
        # Logic Brain: Temperature 0 for strict JSON/Rules
        self.llm_logic = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.0,
            api_key=GROQ_API_KEY
        )
        
        # Chat Brain: Temperature 0.3 for natural conversation
        self.llm_chat = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.3,
            api_key=GROQ_API_KEY
        )

        # 3. Embeddings (Local CPU) - This works fine on Render standard instances
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        
        # 4. Simulate Sleep (Database cleanup)
        try:
            self.db_brain.rpc("decay_memory").execute()
        except: pass
        
        self.session_state = { 
            "last_interaction_id": None, 
            "user_role": "admin",
            "learning_disabled": False
        }
        print(f">> ✅ System Online (Powered by Groq).")

    # --- HELPER: Safe Invocation for Groq ---
    # Groq returns an object (AIMessage), Ollama returned a string.
    # This wrapper extracts the string content safely.
    def _ask_llm(self, llm, prompt: str) -> str:
        try:
            response = llm.invoke(prompt)
            # If response is an object with .content, extract it. If string, return it.
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f">> ⚠️ LLM Error: {e}")
            return ""

    # =========================================================================
    # 🧠 BRAIN 5: GOVERNED SELF-IMPROVEMENT
    # =========================================================================
    def _should_learn(self, intent: str, confidence: float, action_log: str) -> bool:
        if self.session_state.get("learning_disabled", False): return False
        if intent in ["zone_risk_analysis", "risk_analysis", "status_report"]: return False
        if confidence < 0.85: return False
        if intent in ["chat", "greeting", "unknown", "admin_kill_switch"]: return False
        if action_log in ["general_chat", "error", "neutral"]: return False
        print(">> 🟢 Gate Open: Interaction is high-quality.")
        return True

    def learn_reasoning_rule(self, question: str, intent: str, outcome: str):
        try:
            prompt = f"""
            Analyze this interaction and extract a GENERAL RULE for future reasoning.
            Question: "{question}"
            Intent: "{intent}"
            Action: "{outcome}"
            Output ONE short, imperative rule starting with "When...".
            Do not include explanations.
            """
            # Changed to use helper
            rule = self._ask_llm(self.llm_logic, prompt).strip()
            
            if len(rule) < 15 or not rule.lower().startswith("when"): return

            existing = self.db_brain.table("learnings").select("id").eq("topic", "reasoning_rule").eq("insight", rule).execute().data
            if existing:
                self._update_rule_health([existing[0]['id']], "success")
                return

            vec = self.embedder.encode(rule).tolist()
            self.db_brain.table("learnings").insert({
                "topic": "reasoning_rule",
                "insight": rule,
                "embedding": vec
            }).execute()
            print(f">> 🎓 Rule Learned: {rule}")
        except: pass

    def _recall_rules(self, text: str) -> Tuple[str, List[int]]:
        try:
            vec = self.embedder.encode(text).tolist()
            res = self.db_brain.rpc('match_learnings_robust', {
                'query_embedding': vec, 'match_threshold': 0.60, 'match_count': 3
            }).execute()
            valid_rules = [r for r in res.data if "When" in r['insight'] or "Always" in r['insight']]
            if valid_rules:
                return "\n".join([r['insight'] for r in valid_rules]), [r['id'] for r in valid_rules]
        except: pass
        return "", []

    def _update_rule_health(self, rule_ids: List[int], outcome: str):
        if not rule_ids: return
        delta = 0.05 if outcome == "success" else -0.25
        for rid in rule_ids:
            try:
                curr = self.db_brain.table("learnings").select("confidence, usage_count").eq("id", rid).single().execute().data
                new_conf = min(1.5, max(0.0, curr['confidence'] + delta))
                self.db_brain.table("learnings").update({
                    "confidence": new_conf, "usage_count": curr['usage_count'] + 1
                }).eq("id", rid).execute()
            except: pass

    # =========================================================================
    # 🧠 BRAIN 3: TOOL BRAIN (VERIFIED DATA)
    # =========================================================================
    def get_user_360(self, target_user_id: str) -> Dict[str, Any]:
        try:
            try: uuid.UUID(str(target_user_id))
            except ValueError: pass 
            
            incidents = self.db_live.table("incidents").select("*").eq("user_id", target_user_id).execute().data
            locations = self.db_live.table("latest_locations_live").select("*").eq("user_id", target_user_id).limit(50).execute().data
            zones = self.db_live.table("zones").select("*").execute().data
            
            danger_encounters = 0
            for loc in locations:
                lat, lng = self._extract_coords(loc)
                if not lat: continue
                for z in zones:
                    z_lat, z_lng = self._extract_coords(z)
                    if z_lat and self._haversine(lat, lng, z_lat, z_lng) < (z.get('radius', 2000)/1000):
                        danger_encounters += 1

            return {
                "exists": True if (incidents or locations) else False,
                "user_id": target_user_id,
                "stats": {
                    "sos": len([i for i in incidents if i['type'] == 'sos']),
                    "danger": danger_encounters,
                    "safe": max(0, len(locations) - danger_encounters)
                },
                "heatmap": [{"lat": self._extract_coords(l)[0], "lng": self._extract_coords(l)[1]} for l in locations]
            }
        except: return {"exists": False}

    def calculate_risk_ranking(self) -> List[Dict]:
        try:
            sos = self.db_live.table("incidents").select("user_id").eq("type", "sos").neq("status", "resolved").execute().data
            scores = Counter([s['user_id'] for s in sos])
            top_users = scores.most_common(5)
            if not top_users: return []

            user_ids = [u[0] for u in top_users]
            users_data = self.db_live.table("users").select("id, name, email").in_("id", user_ids).execute().data
            name_map = {u['id']: u.get('name', 'Unknown') for u in users_data}
            
            return [{"user_id": uid, "name": name_map.get(uid, "Unknown"), "score": count * 50} for uid, count in top_users]
        except: return []

    def calculate_zone_risk(self) -> Tuple[str, Dict[str, int]]:
        try:
            res = self.db_live.rpc("zone_sos_counts").execute()
            rows = res.data or []

            if not rows:
                return ("No SOS calls detected inside registered danger zones.", {})

            chart_data = {str(row["zone_name"]): int(row["sos_count"]) for row in rows}

            if not chart_data: return ("No active zone data.", {})

            max_val = max(chart_data.values())
            top_zones = [k for k,v in chart_data.items() if v == max_val]
            
            report = f"**🏆 Highest Risk:** {', '.join(top_zones)} ({max_val} SOS)\n\n"
            report += "**🚨 Danger Zone Breakdown:**\n"
            for zone, count in chart_data.items():
                report += f"- **{zone}**: {count} SOS calls\n"

            return (report, chart_data)

        except Exception as e:
            print(f"Zone Risk Error: {e}")
            return ("Unable to compute danger zone risk.", {})

    def get_global_map_data(self) -> List[Dict[str, float]]:
        try:
            data = self.db_live.table("incidents").select("lat, lng").eq("type", "sos").neq("status", "resolved").execute().data
            markers = []
            for row in data:
                lat, lng = self._extract_coords(row)
                if lat: markers.append({"lat": lat, "lng": lng})
            return markers
        except: return []

    def get_weekly_trend_data(self) -> Dict[str, int]:
        try:
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            data = self.db_live.table("incidents").select("created_at").eq("type", "sos").gte("created_at", seven_days_ago).execute().data
            
            counts = Counter()
            for row in data:
                date_str = row['created_at'].split('T')[0]
                counts[date_str] += 1
                
            result = {}
            for i in range(7):
                d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                result[d] = counts.get(d, 0)
            return result
        except: return {}

    def get_live_metrics(self) -> Dict[str, Any]:
        return {
            "active_sos": self._count_by_definition("active_sos"),
            "total_sos": self._count_by_definition("total_sos"),
            "total_users": self._count_rows("users"),
            "solo_users": self._count_by_definition("solo_users"),
            "group_users": self._count_by_definition("group_users"),
            "active_groups": self._count_by_definition("active_groups"),
            "danger_zones": self._count_by_definition("danger_zones"),
        }

    def _log_interaction(self, query: str, intent: str, response: str):
        try:
            self.db_brain.table("interactions").insert({
                "user_query": query,
                "intent": intent,
                "response": str(response)[:1000]
            }).execute()
        except: pass

    # =========================================================================
    # 🧠 BRAIN 1 & 2: PERCEPTION
    # =========================================================================
    def _perceive_intent(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        
        if "disable learning" in text_lower: 
            return {"intent": "admin_kill_switch", "confidence": 1.0}
        
        if ("zone" in text_lower or "chart" in text_lower or "graph" in text_lower) and ("sos" in text_lower or "risk" in text_lower or "call" in text_lower):
            return {"intent": "zone_risk_analysis", "confidence": 1.0} 
            
        if "highest" in text_lower or "most" in text_lower or "top" in text_lower:
            if "risk" in text_lower or "sos" in text_lower or "danger" in text_lower:
                return {"intent": "risk_analysis", "confidence": 1.0}

        if "status" in text_lower or "count" in text_lower or "total" in text_lower or "how many" in text_lower or "number of" in text_lower:
             return {"intent": "status_report", "confidence": 1.0}

        if "details of" in text_lower or "stats for" in text_lower or "about user" in text_lower:
             return {"intent": "user_analytics", "confidence": 1.0}

        prompt = f"""
        Classify query: "user_analytics", "risk_analysis", "zone_risk_analysis", "status_report", "chat".
        Assign confidence (0.0-1.0).
        User: "{text}"
        Format: JSON {{ "intent": "...", "confidence": 0.9 }}
        """
        try:
            # Changed to use helper
            raw = self._ask_llm(self.llm_logic, prompt).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: pass
        return {"intent": "chat", "confidence": 0.0}

    def process_query(self, user_text: str) -> Dict:
        perception = self._perceive_intent(user_text)
        intent = perception.get("intent", "chat")
        confidence = perception.get("confidence", 0.0)
        
        print(f">> 🧠 Perception: {intent} ({confidence})")

        learned_rules_text, active_rule_ids = self._recall_rules(user_text)
        
        response_text = ""
        ui_widget = None
        widget_data = None
        action_log = "general_chat"
        success_signal = "failure" 

        # --- 1. ANALYTICS ---
        if intent == "admin_kill_switch":
            if "auth=delta_force_99" in user_text.lower():
                self.session_state["learning_disabled"] = True
                response_text = "🔒 **SECURITY:** Learning DISABLED."
            else:
                response_text = "❌ **ACCESS DENIED**"
            self._log_interaction(user_text, intent, response_text)
            return {"response": response_text, "ui_widget": None, "widget_data": None}

        if intent == "zone_risk_analysis":
            report, chart_data = self.calculate_zone_risk()
            
            final_widget_data = None
            if chart_data:
                final_widget_data = {
                    "pie_chart": chart_data,
                    "bar_chart": chart_data,
                    "map_data": chart_data
                }
                ui_widget = "zone_stats"

            self._log_interaction(user_text, intent, report)
            return {
                "response": report,
                "ui_widget": ui_widget,
                "widget_data": final_widget_data
            }

        if intent == "risk_analysis":
            scores = self.calculate_risk_ranking()
            if scores:
                response_text = "**High Risk Users:**\n" + "\n".join([f"- **{u['name']}** ({u['user_id']})\n  Risk: {u['score']}" for u in scores])
            else:
                response_text = "No high risk users found."
            self._log_interaction(user_text, intent, response_text)
            return {"response": response_text, "ui_widget": None, "widget_data": None}

        if intent == "status_report":
            m = self.get_live_metrics()
            text_lower = user_text.lower()
            
            if "group" in text_lower:
                response_text = f"👥 **Group Users Count:** {m['group_users']}"
            elif "solo" in text_lower:
                response_text = f"👤 **Solo Users Count:** {m['solo_users']}"
            elif "active" in text_lower and "sos" in text_lower:
                response_text = f"🚨 **Active SOS (Live):** {m['active_sos']}"
            elif "total" in text_lower and "sos" in text_lower:
                response_text = f"📞 **Total SOS (History):** {m['total_sos']}"
            elif "user" in text_lower:
                response_text = f"👥 **Total Users:** {m['total_users']}"
            elif "zone" in text_lower:
                response_text = f"☢️ **Danger Zones:** {m['danger_zones']}"
            else:
                global_map = self.get_global_map_data()
                weekly_trend = self.get_weekly_trend_data()
                
                response_text = f"🛡️ **System Status:**\n- Active SOS: {m['active_sos']}\n- Group Users: {m['group_users']}\n- Danger Zones: {m['danger_zones']}"
                
                # FULL DASHBOARD WIDGET
                self._log_interaction(user_text, intent, response_text)
                return {
                    "response": response_text,
                    "ui_widget": "admin_dashboard_full",
                    "widget_data": {
                        "stats": m,
                        "map_markers": global_map,
                        "line_chart": weekly_trend
                    }
                }
            
            self._log_interaction(user_text, intent, response_text)
            return {"response": response_text, "ui_widget": None, "widget_data": None}

        # --- 2. COMPLEX LOGIC ---
        try:
            if intent == "user_analytics":
                user_match = re.search(r"(user|details)\s+([A-Za-z0-9\-]+)", user_text, re.IGNORECASE)
                if user_match:
                    tid = user_match.group(2).strip()
                    if len(tid) < 3 or tid.lower() in ["the", "has", "who", "is"]:
                        response_text = "Please specify a valid User ID."
                    else:
                        data = self.get_user_360(tid)
                        if data["exists"]:
                            summary_prompt = f"GOVERNANCE: {learned_rules_text}\nDATA: {data['stats']}\nWrite profile."
                            # Changed to use helper
                            response_text = self._ask_llm(self.llm_chat, summary_prompt)
                            ui_widget = "dashboard_stats"
                            widget_data = {"pie_chart": {"Safe": data['stats']['safe'], "Danger": data['stats']['danger']}, "heatmap": data['heatmap']}
                            action_log = "fetched_user_profile"
                            success_signal = "success"
                        else:
                            response_text = f"User '{tid}' not found."
                            action_log = "error"
                else: response_text = "Provide ID."
            
            else:
                # CHAT LOGIC
                chat_prompt = f"""
                SYSTEM: You are 'Suraksha Sentinel', a Safety AI.
                CONTEXT: {learned_rules_text}
                USER: {user_text}
                Answer strictly about safety. If asked about SOS, it means Emergency.
                """
                # Changed to use helper
                response_text = self._ask_llm(self.llm_chat, chat_prompt)
                success_signal = "neutral"

        except Exception as e:
            response_text = f"Error: {str(e)}"
            action_log = "error"
            success_signal = "failure"

        # --- 3. MAINTENANCE ---
        self._log_interaction(user_text, intent, response_text)

        if active_rule_ids and success_signal != "neutral":
            self._update_rule_health(active_rule_ids, success_signal)

        if self._should_learn(intent, confidence, action_log):
            self.learn_reasoning_rule(user_text, intent, action_log)

        return {
            "response": response_text,
            "ui_widget": ui_widget,
            "widget_data": widget_data
        }

    # --- HELPERS ---
    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        a = math.sin(math.radians(lat2-lat1)/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2-lon1)/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _extract_coords(self, row: Dict) -> Tuple[Optional[float], Optional[float]]:
        try:
            if 'lat' in row and row['lat']: return float(row['lat']), float(row['lng'])
            if 'latitude' in row and row['latitude']: return float(row['latitude']), float(row['longitude'])
            if 'geometry' in row and isinstance(row['geometry'], str):
                match = re.search(r'POINT\s*\(?([-0-9\.]+)\s+([-0-9\.]+)\)?', row['geometry'], re.IGNORECASE)
                if match:
                    lng, lat = map(float, match.groups())
                    return lat, lng 
            if 'coordinates' in row and isinstance(row['coordinates'], list):
                return float(row['coordinates'][1]), float(row['coordinates'][0])
        except Exception: pass
        return None, None

    def _count_by_definition(self, key):
        d = DEFINITIONS[key]
        return self._count_rows(d["table"], d["filters"])

    def _count_rows(self, table, filters=[]):
        try:
            q = self.db_live.table(table).select("*", count="exact", head=True)
            for c, o, v in filters:
                if o == "eq": q = q.eq(c, v)
                elif o == "neq": q = q.neq(c, v)
                elif o == "in": q = q.in_(c, v)
                elif o == "is": q = q.is_(c, v)
                elif o == "not.is": q = q.not_.is_(c, v)
            return q.execute().count or 0
        except: return 0

# =========================================================================
# 🔌 API
# =========================================================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = SentinelEngine()

class Query(BaseModel): text: str

@app.post("/chat")
def chat_endpoint(q: Query):
    res = engine.process_query(q.text)
    return {"response": res.get("response", ""), "ui_widget": res.get("ui_widget"), "widget_data": res.get("widget_data")}