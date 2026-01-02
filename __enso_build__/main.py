
import os, json, time, sys
import requests
from typing import Any, List, Dict, Literal 
from pydantic import BaseModel
from abc import ABC, abstractmethod

MOCKS = {}

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
        
        url = "[https://api.openai.com/v1/chat/completions](https://api.openai.com/v1/chat/completions)"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # Force JSON via prompt + mode
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
        
        # Call Google API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
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
        # Robust cleaner for Markdown/Backticks
        raw = raw.strip()
        if raw.startswith("```json"): raw = raw[7:]
        elif raw.startswith("```"): raw = raw[3:]
        if raw.endswith("```"): raw = raw[:-3]
        return raw.strip()

    def run(self, input_text, response_model):
        if self.name in MOCKS:
            print(f"    [Mock] Serving response for '{self.name}'")
            return Probabilistic(value=MOCKS[self.name], confidence=1.0, cost=0.0, model_used="MOCK")

        print(f"\n[Enso] Agent '{self.name}' -> {self.model} ({self.spec.get('type')})...")
        schema = response_model.model_json_schema()
        
        start_t = time.time()
        # 1. Call API
        raw_json = self.driver.complete(self.model, self.instruction, input_text, schema)
        latency = time.time() - start_t
        
        # 2. Clean Response
        clean_json = self._clean_json(raw_json)
        
        in_tok = len(self.instruction)//4 + len(input_text)//4
        out_tok = len(clean_json)//4
        cost = (in_tok/1e6 * self.spec.get('cost_in', 0)) + (out_tok/1e6 * self.spec.get('cost_out', 0))

        try:
            # 3. Parse & Validate
            data = json.loads(clean_json)
            val = response_model(**data)
            print(f"    [Meta] Cost: ${round(cost, 6)} | Latency: {round(latency, 2)}s")
            return Probabilistic(value=val, confidence=0.99, cost=cost, model_used=self.model)
        except Exception as e:
            print(f"    [Parse Error] {e}")
            print(f"    [Raw Output] {raw_json[:100]}...") # Print first 100 chars
            return Probabilistic(value=None, confidence=0.0, cost=0.0, model_used=self.model)

def run_tests(include_ai=False):
    print(f"\n🧪 Running Tests (Include AI: {include_ai})...")
    g = globals()
    tests = [name for name in g if name.startswith("test_")]
    passed, skipped = 0, 0
    for t in tests:
        if "_AI_" in t and not include_ai:
            skipped += 1; continue
        MOCKS.clear()
        try:
            g[t](); print(f"✅ PASS: {t}"); passed += 1
        except Exception as e: print(f"❌ FAIL: {t} ({e})")
    print(f"\nSummary: {passed} Passed, {skipped} Skipped.")


class Sentiment(BaseModel):
    mood: str
    score: int


def analyze(text: str):
    agent = EnsoAgent(name="analyze", instruction="Analyze the sentiment.", model="gemini-flash-latest")
    return agent.run(str(locals()), Sentiment)


print("Running Analysis...")

result = analyze("I completely love this new programming language!")

print(result.value.mood)

print(result.value.score)