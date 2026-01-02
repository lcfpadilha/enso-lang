import sys
from lark import Lark, Transformer

# ==========================================
# 1. THE GRAMMAR (The Laws of Ensō)
# ==========================================
enso_grammar = r"""
    start: (struct_def | ai_fn_def | test_def | statement)*

    // --- Definitions ---
    struct_def: "struct" NAME "{" field_def* "}"
    field_def: NAME ":" type_expr [","]

    ai_fn_def: "ai" "fn" NAME "(" arg_list? ")" "->" type_expr "{" ai_body "}"
    ai_body: instruction_stmt config_stmt* validate_block?
    
    // --- Testing ---
    test_def: "test" [test_type] STRING "{" test_body "}"
    test_type: "ai"
    test_body: (mock_stmt | statement | assertion)*
    
    mock_stmt: "mock" NAME "=>" expr ";"
    assertion: "assert" expr ";"

    // --- Details ---
    instruction_stmt: "instruction" ":" STRING
    config_stmt: CONFIG_KEY ":" (NUMBER | STRING) [","]
    CONFIG_KEY: "temperature" | "model"
    validate_block: "validate" "(" NAME ")" "{" validation_rule* "}"
    validation_rule: expr "=>" STRING [","]

    // --- TYPE EXPRESSIONS (FIXED) ---
    // We strictly separate the rules so the Transformer knows which branch was hit
    type_expr: list_type | enum_type | simple_type
    
    simple_type: NAME
    list_type: "List" "<" type_expr ">"
    enum_type: "Enum" "<" (STRING [","])* ">"

    arg_list: arg_def ("," arg_def)*
    arg_def: NAME ":" type_expr

    // --- Statements ---
    statement: let_stmt | print_stmt
    let_stmt: "let" NAME "=" expr ";"
    print_stmt: "print" "(" expr ")" ";"
    
    // --- Expressions ---
    ?expr: NAME | prop_access | binary_expr | struct_init | call_expr | NUMBER | STRING
    
    struct_init: NAME "{" (field_init)* "}"
    field_init: NAME ":" expr [","]

    call_expr: NAME "(" args_call? ")"
    args_call: expr ("," expr)*
    
    prop_access: NAME ("." NAME)+

    binary_expr: expr OPERATOR expr
    OPERATOR: ">" | "<" | "==" | "!=" | ">=" | "<=" | "&&" | "||"

    NAME: /[a-zA-Z_]\w*/
    STRING: /".*?"/
    NUMBER: /\d+(\.\d+)?/
    %import common.WS
    %ignore WS
"""

# ==========================================
# 2. THE RUNTIME PREAMBLE
# ==========================================
RUNTIME_PREAMBLE = """
import os, json, time, sys
import requests
from typing import Any, List, Dict, Literal 
from pydantic import BaseModel, ValidationError
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
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        system_prompt = f"{system}\\n\\nYou MUST return valid JSON matching this schema:\\n{json.dumps(schema)}"
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
            "contents": [{"parts": [{"text": f"System: {system}\\nUser: {user}"}]}],
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
            "prompt": f"{system}\\nSchema: {json.dumps(schema)}\\nUser: {user}", 
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

        print(f"\\n[Enso] Agent '{self.name}' -> {self.model}...")
        schema = response_model.model_json_schema()

        start_t = time.time()
        raw_json = self.driver.complete(self.model, self.instruction, input_text, schema)
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
    print(f"\\n🧪 Running Tests (Include AI: {include_ai})...")
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
    print(f"\\nSummary: {passed} Passed, {skipped} Skipped.")
"""

# ==========================================
# 3. THE TRANSPILER
# ==========================================
class EnsoTransformer(Transformer):
    def start(self, items): return RUNTIME_PREAMBLE + "\n\n" + "\n\n".join(items)
    
    def struct_def(self, args):
        name, *fields = args
        return f"class {name}(BaseModel):\n    " + "\n    ".join(fields)
    def field_def(self, args): return f"{args[0]}: {args[1]}"

    def ai_fn_def(self, args):
        if len(args) == 4: arg_str, ret, body = args[1], args[2], args[3]
        else: arg_str, ret, body = "", args[1], args[2]
        instr = body['instruction']
        model = body.get('model', '"gpt-4o"')
        name = args[0]
        
        return f"""
def {name}({arg_str}):
    agent = EnsoAgent(name="{name}", instruction={instr}, model={model})
    # Filter variables to only send arguments, not the agent itself
    inputs = {{k:v for k,v in locals().items() if k != 'agent' and not k.startswith('__')}}
    return agent.run(str(inputs), {ret})
"""

    def ai_body(self, args):
        d = {}
        for x in args: 
            if isinstance(x, dict): d.update(x)
        return d
    
    def test_def(self, args):
        if len(args) == 3:
            is_ai = True
            name = args[1].replace('"', '')
            body = args[2]
        else:
            is_ai = False
            name = args[0].replace('"', '')
            body = args[1]
        slug = name.replace(" ", "_")
        prefix = "test_AI_" if is_ai else "test_"
        return f"def {prefix}{slug}():\n    pass\n    {body}"

    def test_body(self, args): return "\n    ".join(args)
    def call_expr(self, args):
        name = args[0]
        params = args[1] if len(args) > 1 else ""
        return f"{name}({params})"
    
    def args_call(self, args): return ", ".join(args)
    def mock_stmt(self, args): return f"MOCKS['{args[0]}'] = {args[1]}"
    def assertion(self, args): return f"assert {args[0]}"
    
    def statement(self, args): return args[0]
    def print_stmt(self, args): return f"print({args[0]})"
    def let_stmt(self, args): return f"{args[0]} = {args[1]}"
    def prop_access(self, args): return ".".join(args)

    def instruction_stmt(self, args): return {"instruction": args[0]}
    def config_stmt(self, args): return {args[0]: args[1]}
    def arg_list(self, args): return ", ".join(args)
    def arg_def(self, args): return f"{args[0]}: {args[1]}"
    
    def type_expr(self, args): return args[0] # Pass-through for the one chosen child

    def simple_type(self, args):
        arg0 = str(args[0])
        m = {"String":"str", "Int":"int", "Float":"float"}
        return m.get(arg0, arg0)

    def list_type(self, args):
        return f"List[{args[0]}]"

    def enum_type(self, args):
        options = ", ".join([str(x) for x in args])
        return f"Literal[{options}]"

    def struct_init(self, args): return f"{args[0]}({', '.join(args[1:])})"
    def field_init(self, args): return f"{args[0]}={args[1]}"
    def binary_expr(self, args): return f"{args[0]} {args[1]} {args[2]}"
    def NAME(self, t): return str(t)
    def STRING(self, t): return str(t)
    def NUMBER(self, t): return str(t)
    def CONFIG_KEY(self, t): return str(t)
    def OPERATOR(self, t): return str(t)

# ==========================================
# 4. INTROSPECTION
# ==========================================
class SchemaExtractor(Transformer):
    def start(self, items): return [i for i in items if isinstance(i, dict) and i.get('type') == 'function']
    def ai_fn_def(self, args):
        if len(args) == 4: arg_str, ret, body = args[1], args[2], args[3]
        else: arg_str, ret, body = "", args[1], args[2]
        arg_list = []
        if arg_str:
            for part in arg_str.split(","):
                p = part.split(":")
                arg_list.append({"name": p[0].strip(), "type": p[1].strip()})
        return {"type": "function", "name": args[0], "args": arg_list, "return": ret}
    def struct_def(self, args): return None
    def test_def(self, args): return None
    def statement(self, args): return None
    def print_stmt(self, args): return None
    def let_stmt(self, args): return None
    def arg_list(self, args): return ", ".join(args)
    def arg_def(self, args): return f"{args[0]}: {args[1]}"
    def type_expr(self, args): return "Any"
    def NAME(self, t): return str(t)
    def STRING(self, t): return str(t)
    def NUMBER(self, t): return str(t)
    def ai_body(self, t): return None
    def instruction_stmt(self, t): return None
    def config_stmt(self, t): return None
    def validate_block(self, t): return None
    def call_expr(self, t): return None
    def args_call(self, t): return None
    def struct_init(self, t): return None
    def field_init(self, t): return None
    def binary_expr(self, t): return None
    def prop_access(self, t): return None
    def test_body(self, t): return None
    def mock_stmt(self, t): return None
    def assertion(self, t): return None

def compile_source(source_code):
    parser = Lark(enso_grammar, parser='lalr', transformer=EnsoTransformer())
    return parser.parse(source_code)

def analyze_source(source_code):
    parser = Lark(enso_grammar, parser='lalr', transformer=SchemaExtractor())
    try: return parser.parse(source_code)
    except Exception: return []