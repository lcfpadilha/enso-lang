
import os, json, time, sys
from typing import Any, List, Dict
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
    def complete(self, system, user, schema) -> str: pass

class OpenAIDriver(LLMDriver):
    def complete(self, system, user, schema):
        time.sleep(0.5)
        return '{"mood": "Happy", "score": 8}'

class GeminiDriver(LLMDriver):
    def complete(self, system, user, schema):
        time.sleep(0.3)
        return '{"mood": "Ecstatic", "score": 10}'

class LocalDriver(LLMDriver):
    def complete(self, system, user, schema):
        return '{"mood": "Neutral", "score": 5}'

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

    def run(self, input_text, response_model):
        if self.name in MOCKS:
            print(f"    [Mock] Serving response for '{self.name}'")
            return Probabilistic(value=MOCKS[self.name], confidence=1.0, cost=0.0, model_used="MOCK")

        print(f"\n[Enso] Agent '{self.name}' -> {self.model} ({self.spec.get('type')})...")
        schema = response_model.model_json_schema()
        raw_json = self.driver.complete(self.instruction, input_text, schema)
        
        in_tok = len(self.instruction)//4 + len(input_text)//4
        out_tok = len(raw_json)//4
        cost_in = self.spec.get('cost_in', 0)
        cost_out = self.spec.get('cost_out', 0)
        cost = (in_tok/1e6 * cost_in) + (out_tok/1e6 * cost_out)

        try:
            data = json.loads(raw_json)
            val = response_model(**data)
            print(f"    [Meta] Cost: ${round(cost, 6)}")
            return Probabilistic(value=val, confidence=0.99, cost=cost, model_used=self.model)
        except Exception as e:
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


class Sentiment(BaseModel):
    mood: str
    score: int


def analyze(text: str):
    agent = EnsoAgent(name="analyze", instruction="Analyze sentiment.", model="gemini-flash-latest")
    return agent.run(str(locals()), Sentiment)


print("Ensō Initialized.")

print(analyze("very nice!"))