
import os, json, time, sys
import requests
from typing import Any, List, Dict, Literal 
from pydantic import BaseModel, ValidationError
from abc import ABC, abstractmethod

MOCKS = {}

def resolve_refs(schema):
    defs = schema.get('$defs', {})

    def expand(node):
        if isinstance(node, dict):
            # If it's a reference, replace it with the actual definition
            if '$ref' in node:
                ref_name = node['$ref'].split('/')[-1]
                # Recursively expand the definition we found
                return expand(defs[ref_name])
            
            # Otherwise, traverse dict, removing '$defs' keys
            return {
                k: expand(v) 
                for k, v in node.items() 
                if k != '$defs'
            }
        elif isinstance(node, list):
            return [expand(item) for item in node]
        else:
            return node

    return expand(schema)

# --- Dynamic Configuration ---
def load_model_config(model_name):
    try:
        paths = ["models.json", os.path.expanduser("~/.enso/models.json")]
        registry = {}
        for p in paths:
            if os.path.exists(p):
                with open(p, "r") as f:
                    registry = json.load(f)
                break
        if model_name in registry:
            return registry[model_name]
        return {"type": "openai", "cost_in": 0, "cost_out": 0}
    except Exception:
        return {"type": "openai", "cost_in": 0, "cost_out": 0}

# --- Drivers ---
class LLMDriver(ABC):
    @abstractmethod
    def complete(self, model: str, system: str, user: str, schema: Dict) -> str: pass

class OpenAIDriver(LLMDriver):
    def complete(self, model: str, system: str, user: str, schema: Dict) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: raise ValueError("Missing OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        system_prompt = f"{system}\n\nYou MUST return valid JSON matching this schema:\n{json.dumps(schema)}"
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        try:
            resp = requests.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            if 'resp' in locals(): print(f"    [Details] {resp.text}")
            raise e

class GeminiDriver(LLMDriver):
    def complete(self, model: str, system: str, user: str, schema: Dict) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key: raise ValueError("Missing GEMINI_API_KEY")
        api_key = api_key.strip()
        clean_model = model.replace('"', '').replace("'", "").strip()
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": f"System: {system}\nUser: {user}"}]}],
            "generationConfig": {
                "response_mime_type": "application/json", 
                "response_schema": schema
            }
        }

        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"    [Error] Gemini Connection Failed: {url}")
            if 'resp' in locals(): print(f"    [Details] {resp.text}")
            raise e

class LocalDriver(LLMDriver):
    def complete(self, model: str, system: str, user: str, schema: Dict) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3", 
            "prompt": f"{system}\nSchema: {json.dumps(schema)}\nUser: {user}", 
            "format": "json", 
            "stream": False
        }
        resp = requests.post(url, json=payload)
        return resp.json()['response']

def get_driver(driver_type):
    if driver_type == "gemini": return GeminiDriver()
    if driver_type == "local": return LocalDriver()
    return OpenAIDriver()

# --- Agent ---
class Probabilistic(BaseModel):
    value: Any
    confidence: float
    cost: float
    model_used: str

class EnsoAgent:
    def __init__(self, name, instruction, model="gpt-4o"):
        self.name = name
        self.instruction = instruction
        self.model = model
        self.spec = load_model_config(model)
        self.driver = get_driver(self.spec.get('type', 'openai'))

    def _clean_json(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```json"): raw = raw[7:]
        elif raw.startswith("```"): raw = raw[3:]
        if raw.endswith("```"): raw = raw[:-3]
        if raw.startswith("System:"): raw = raw.replace("System:", "", 1)
        return raw.strip()

    def run(self, input_text, response_model):
        if self.name in MOCKS:
            print(f"    [Mock] Serving response for '{self.name}'")
            return Probabilistic(value=MOCKS[self.name], confidence=1.0, cost=0.0, model_used="MOCK")

        print(f"\n[Enso] Agent '{self.name}' -> {self.model}...")
        raw_schema = response_model.model_json_schema()
        clean_schema = resolve_refs(raw_schema)

        start_t = time.time()
        raw_json = self.driver.complete(self.model, self.instruction, input_text, clean_schema)
        latency = time.time() - start_t
        
        clean_json = self._clean_json(raw_json)
        in_tok = len(self.instruction)//4 + len(input_text)//4
        out_tok = len(clean_json)//4
        cost = (in_tok/1e6 * self.spec.get('cost_in', 0)) + (out_tok/1e6 * self.spec.get('cost_out', 0))

        try:
            data = json.loads(clean_json)
            val = response_model(**data)
            print(f"    [Meta] Cost: ${round(cost, 6)} | Latency: {round(latency, 2)}s")
            return Probabilistic(value=val, confidence=0.99, cost=cost, model_used=self.model)
        except Exception as e:
            print(f"    [Parse Error] {e}")
            print(f"    [Raw Output] {raw_json[:200]}...") 
            return Probabilistic(value=None, confidence=0.0, cost=0.0, model_used=self.model)

# --- Test Runner ---
def run_tests(include_ai=False):
    print(f"\n🧪 Running Tests (Include AI: {include_ai})...")
    g = globals()
    tests = [name for name in g if name.startswith("test_")]
    passed = 0
    skipped = 0
    for t in tests:
        is_ai_test = "_AI_" in t
        print(t)
        if is_ai_test and not include_ai:
            skipped += 1
            continue
        MOCKS.clear()
        try:
            g[t]()
            print(f"✅ PASS: {t}")
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {t} ({e})")
    print(f"\nSummary: {passed} Passed, {skipped} Skipped.")


class Skill(BaseModel):
    name: str
    level: Literal["Beginner", "Intermediate", "Expert"]

class Candidate(BaseModel):
    name: str
    years_experience: int
    top_skills: List[Skill]
    last_role: str

class Evaluation(BaseModel):
    hiring_decision: Literal["Hire", "Reject", "Interview"]
    fit_score: int
    reasoning: str

class FinalDecision(BaseModel):
    decision: Literal["First Candidate", "Second Candidate", "None"]


def parse_resume(raw_text: str):
    agent = EnsoAgent(name="parse_resume", instruction="Extract candidate details. Be strict with skill levels.", model="gemini-2.5-flash-lite")
    # Filter variables to only send arguments, not the agent itself
    inputs = {k:v for k,v in locals().items() if k != 'agent' and not k.startswith('__')}
    return agent.run(str(inputs), Candidate)



def compare_candidates(candidate_a: Candidate, candidate_b: Candidate):
    agent = EnsoAgent(name="compare_candidates", instruction="You are a hiring manager looking for a Senior Python Engineer. Evaluate first candidate over the second one. Returns the best candidate, IF ANY, to the job.", model="gemini-2.5-flash-lite")
    # Filter variables to only send arguments, not the agent itself
    inputs = {k:v for k,v in locals().items() if k != 'agent' and not k.startswith('__')}
    return agent.run(str(inputs), FinalDecision)


candidates = ["SARAH CONNOR  \nLos Angeles, CA • 555-010-9988 • sarah.connor@sky.net  \n\nPROFILE  \nExperienced Data Scientist specialized in predictive modeling and anomaly detection.  \n\nWORK HISTORY  \nCyberdyne Systems — Lead Data Scientist  \n03/2019 – Current  \n• Built neural networks for image recognition using TensorFlow and Keras. • Optimized SQL queries reducing data retrieval time by 50%. • Collaborated with product teams to define data requirements.  \n\nTechCom | Data Analyst  \n06/2015 – 02/2019  \nAnalyzed large datasets to identify market trends... Automated reporting dashboards using Tableau and Python (Pandas).  \n\nEDUCATION  \nM.S. Statistics — University of West Coast (2015)  \nB.S. Mathematics — University of West Coast (2013)  \n\nTECHNICAL SKILLS  \nPython, R, SQL, Machine Learning, NLP, BigQuery, Spark, Git", "Alex Murphy\nDetroit, MI\nalex.murphy@ocp.example.edu\nGitHub: github.com/robocop\n\nEDUCATION\nUniversity of Detroit\nBachelor of Science in Computer Engineering\nExpected Graduation: Dec 2024\nGPA: 3.8/4.0\n\nRelevant Coursework: \nData Structures & Algorithms, Operating Systems, Embedded Systems, Computer Vision, Linear Algebra.\n\nPROJECTS\nAutonomous Drone Navigation (Capston Project)\n- Fall 2023\n- Developed C++ control logic for obstacle avoidance using LiDAR data.\n- Implemented A* search algorithm for path planning.\n\nPersonal Portfolio Website\n- Summer 2022\n- Built using React.js and Node.js deployed on Vercel.\n\nSKILLS\nC++, C, Python, MATLAB, Linux/Unix, Git, Verilog.", "Ellen Ripley\nNostromo, Space | 555-012-3456 | ripley@weyland.example.com\n\nProfessional Summary\nI am a dedicated Project Manager with over 10 years of experience leading cross-functional teams in high-stress environments. My focus is on risk management and operational safety.\n\nProfessional Experience\nFrom 2015 to 2023, I served as a Warrant Officer at Weyland-Yutani Corp. In this role, I was responsible for overseeing the commercial towing vehicle 'Nostromo'. My duties included managing a crew of seven, ensuring adherence to safety protocols, and handling crisis management situations. I successfully navigated complex logistical challenges and maintained operational integrity under extreme pressure.\n\nPrior to that, between 2010 and 2015, I worked as a Flight Officer. I coordinated flight paths and managed fuel consumption metrics, resulting in a 15% reduction in operational costs.\n\nEducation\nI hold a Master of Engineering from the New York Aeronautics Institute, achieved in 2009.\n\nCertifications\nPMP (Project Management Professional), Scum Master Certified, Industrial Safety Specialist."]

hired_candidate = Candidate(name="", years_experience=0, last_role="", top_skills=[])

for raw_candidate in candidates:
    candidate = parse_resume(raw_candidate)
    comparison = compare_candidates(candidate, hired_candidate)
    if comparison.value.decision == "First Candidate":
        hired_candidate = candidate

print("Hired candidate is: ")

print(hired_candidate.value.name)