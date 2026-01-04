import sys
import os
import re
from lark import Lark, Transformer

# ==========================================
# 0. THE BUNDLER (New Feature)
# ==========================================
def bundle(file_path, visited=None):
    """
    Recursively replaces 'import "lib.enso";' with the content of that file.
    Acts like a C pre-processor.
    """
    if visited is None: visited = set()
    
    # Resolve absolute path to handle nested imports correctly
    abs_path = os.path.abspath(file_path)
    
    if abs_path in visited:
        return f"// Cycle detected: {file_path} skipped"
    visited.add(abs_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Import not found: {abs_path}")

    with open(abs_path, 'r') as f:
        code = f.read()

    base_dir = os.path.dirname(abs_path)

    def replacer(match):
        # Extract filename from: import "foo.enso";
        rel_path = match.group(1)
        full_path = os.path.join(base_dir, rel_path)
        return f"\n{bundle(full_path, visited)}\n"

    # Regex to find: import "anything";
    # We remove the line and replace it with the file content
    bundled_code = re.sub(r'import\s+"([^"]+)"\s*;', replacer, code)
    
    return bundled_code

# ==========================================
# 1. THE GRAMMAR (The Laws of Ensō)
# ==========================================
enso_grammar = r"""
    start: (import_def | struct_def | ai_fn_def | regular_fn_def | test_def | statement)*

    // --- Imports ---
    import_def: "import" STRING ";"

    // --- Definitions ---
    struct_def: "struct" NAME "{" field_def* "}"
    field_def: NAME ":" type_expr [","]

    // AI Function (Declarative)
    ai_fn_def: "ai" "fn" NAME "(" arg_list? ")" "->" type_expr "{" ai_body "}"
    ai_body: instruction_stmt config_stmt* validate_block?

    // Regular Function (Imperative)
    regular_fn_def: "fn" NAME "(" arg_list? ")" ["->" type_expr] "{" stmt_block "}"
    stmt_block: statement*

    // --- Testing ---
    test_def: "test" [test_type] STRING "{" test_body "}"
    test_type: "ai"
    test_body: (mock_stmt | statement)*
    
    mock_stmt: "mock" NAME "=>" expr ";"

    // --- Details ---
    instruction_stmt: "instruction" ":" STRING [","]
    config_stmt: CONFIG_KEY ":" (NUMBER | STRING) [","]
    CONFIG_KEY: "temperature" | "model"
    validate_block: "validate" "(" NAME ")" "{" validation_rule* "}"
    validation_rule: expr "=>" STRING [","]

    // --- Types ---
    type_expr: result_type | list_type | enum_type | simple_type
    simple_type: NAME | TYPE_NAME
    list_type: "List" "<" type_expr ">"
    enum_type: "Enum" "<" (STRING [","])* ">"
    result_type: "Result" "<" type_expr "," type_expr ">"

    arg_list: arg_def ("," arg_def)*
    arg_def: NAME ":" type_expr

    // --- Statements ---
    statement: let_stmt | assign_stmt | print_stmt | return_stmt | if_stmt | for_stmt | match_stmt | concurrent_stmt | assertion | expr_stmt

    let_stmt: "let" NAME "=" expr ";"
    concurrent_stmt: "concurrent" call_expr "for" NAME "in" expr ";"
    assign_stmt: NAME "=" expr ";"
    print_stmt: "print" "(" expr ")" ";"
    return_stmt: "return" expr ";"
    if_stmt: "if" expr "{" stmt_block "}" ("else" "{" stmt_block "}")?
    for_stmt: "for" NAME "in" expr "{" stmt_block "}"
    match_stmt: "match" expr "{" match_arm* "}"
    match_arm: match_pattern "=>" "{" stmt_block "}" [","]
    match_pattern: "Ok" "(" NAME ")" -> ok_pattern | "Err" "(" NAME ")" -> err_pattern
    assertion: "assert" expr ";"
    expr_stmt: expr ";"
    
    // --- Expressions ---
    ?expr: term (OPERATOR term)*
    ?term: atom | "(" expr ")"
    ?atom: NAME | STRING | NUMBER | prop_access | call_expr | struct_init | list_literal

    list_literal: "[" (expr ("," expr)*)? "]"

    struct_init: NAME "{" (field_init)* "}"
    field_init: NAME ":" expr [","]

    call_expr: NAME "(" args_call? ")"
    args_call: expr ("," expr)*
    prop_access: NAME ("." NAME)+

    // --- Terminals ---
    TYPE_NAME: /[A-Z][a-zA-Z_]\w*/
    NAME: /(?!(?:let|print|return|if|else|for|in|import|struct|ai|fn|test|mock|assert|instruction|temperature|model|validate|List|Enum|Result|match|Ok|Err|concurrent)\b)[a-zA-Z_]\w*/
    STRING: /"(?:[^"\\]|\\.)*"/
    NUMBER: /\d+(\.\d+)?/
    
    OPERATOR: "==" | "!=" | ">=" | "<=" | "&&" | "||" | "+" | "-" | ">" | "<"
    
    COMMENT_1: /\/\/[^\n]*/
    COMMENT_2: /\#[^\n]*/
    BLOCK_COMMENT: /\/\*[\s\S]*?\*\//

    %import common.WS
    %ignore WS
    %ignore COMMENT_1
    %ignore COMMENT_2
    %ignore BLOCK_COMMENT
"""

# ==========================================
# 2. THE RUNTIME PREAMBLE
# ==========================================
RUNTIME_PREAMBLE = """
import os, json, time, sys, asyncio
import requests
from typing import Any, List, Dict, Literal, Union, Optional
from pydantic import BaseModel, ValidationError
from abc import ABC, abstractmethod
from enum import Enum

MOCKS = {}

# ==========================================
# ERROR HANDLING: Result<T, E> Model
# ==========================================

class ErrorKind(str, Enum):
    # Categorizes AI failures for composable error handling
    API_ERROR = "ApiError"
    PARSE_ERROR = "ParseError"
    HALLUCINATION_ERROR = "HallucinationError"
    TIMEOUT_ERROR = "TimeoutError"
    REFUSAL_ERROR = "RefusalError"
    INVALID_CONFIG_ERROR = "InvalidConfigError"

class AIError(BaseModel):
    # Rich error context for AI operations
    kind: ErrorKind
    message: str
    details: Optional[str] = None
    cost: float
    model: str
    timestamp: Optional[str] = None

class Result:
    # Base class for Result<T, E> (Ok or Err)
    def is_ok(self) -> bool:
        return isinstance(self, Ok)
    
    def is_err(self) -> bool:
        return isinstance(self, Err)
    
    def unwrap(self):
        # Get value or raise error
        if isinstance(self, Ok):
            return self.value
        else:
            raise RuntimeError(f"Called unwrap() on Err: {self.error.message}")
    
    def unwrap_or(self, default):
        # Get value or return default
        if isinstance(self, Ok):
            return self.value
        else:
            return default

class Ok(Result):
    # Success case: Ok<T>
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"Ok({self.value})"

class Err(Result):
    # Error case: Err<E>
    def __init__(self, error):
        self.error = error
    
    def __repr__(self):
        return f"Err({self.error})"

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

    def run(self, input_text, response_model) -> Union[Ok, Err]:
        # Execute AI function and return Result<Probabilistic<T>, AIError>.
        # Categorizes all failures as AIError with proper ErrorKind.
        cost = 0.0
        try:
            # Check mocks first
            if self.name in MOCKS:
                print(f"    [Mock] Serving response for '{self.name}'")
                mocked_value = MOCKS[self.name]
                return Ok(Probabilistic(value=mocked_value, confidence=1.0, cost=0.0, model_used="MOCK"))

            # Validate configuration
            if not self.model:
                return Err(AIError(
                    kind=ErrorKind.INVALID_CONFIG_ERROR,
                    message="Model name is empty",
                    cost=0.0,
                    model=self.model or "unknown"
                ))

            print(f"\\n[Enso] Agent '{self.name}' -> {self.model}...")
            raw_schema = response_model.model_json_schema()
            clean_schema = resolve_refs(raw_schema)

            # Call driver with error handling
            try:
                start_t = time.time()
                raw_json = self.driver.complete(self.model, self.instruction, input_text, clean_schema)
                latency = time.time() - start_t
            except ValueError as e:
                # Invalid config (missing API keys, etc)
                return Err(AIError(
                    kind=ErrorKind.INVALID_CONFIG_ERROR,
                    message=str(e),
                    cost=0.0,
                    model=self.model
                ))
            except TimeoutError as e:
                return Err(AIError(
                    kind=ErrorKind.TIMEOUT_ERROR,
                    message="Request timeout",
                    details=str(e),
                    cost=cost,
                    model=self.model
                ))
            except Exception as e:
                # Network/API errors
                return Err(AIError(
                    kind=ErrorKind.API_ERROR,
                    message="API request failed",
                    details=str(e),
                    cost=cost,
                    model=self.model
                ))
            
            # Parse response
            clean_json = self._clean_json(raw_json)
            in_tok = len(self.instruction)//4 + len(input_text)//4
            out_tok = len(clean_json)//4
            cost = (in_tok/1e6 * self.spec.get('cost_in', 0)) + (out_tok/1e6 * self.spec.get('cost_out', 0))

            # Try to parse JSON
            try:
                data = json.loads(clean_json)
            except json.JSONDecodeError as e:
                return Err(AIError(
                    kind=ErrorKind.PARSE_ERROR,
                    message="Invalid JSON response",
                    details=f"{str(e)}\\nRaw: {raw_json[:200]}",
                    cost=cost,
                    model=self.model
                ))
            
            # Try to validate against schema
            try:
                val = response_model(**data)
                print(f"    [Meta] Cost: ${round(cost, 6)} | Latency: {round(latency, 2)}s")
                return Ok(Probabilistic(value=val, confidence=0.99, cost=cost, model_used=self.model))
            except ValidationError as e:
                return Err(AIError(
                    kind=ErrorKind.HALLUCINATION_ERROR,
                    message="Response doesn't match schema",
                    details=str(e),
                    cost=cost,
                    model=self.model
                ))
        
        except Exception as e:
            # Catch-all for unexpected errors
            return Err(AIError(
                kind=ErrorKind.API_ERROR,
                message="Unexpected error",
                details=str(e),
                cost=cost,
                model=self.model
            ))

    async def run_async(self, input_text, response_model) -> Union[Ok, Err]:
        # Async wrapper for concurrent execution
        return self.run(input_text, response_model)

    def run_concurrent(self, items, func_name, response_model):
        # Execute agent.run() for each item concurrently using asyncio
        # Returns List[Result[T, E]] (list of Ok or Err for each item)
        async def gather_results():
            tasks = [self.run_async(item, response_model) for item in items]
            return await asyncio.gather(*tasks)
        
        # Run the async function directly with asyncio.run()
        return asyncio.run(gather_results())

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
    def start(self, items):
        # Filter out empty strings from bundled imports
        items = [i for i in items if i and i.strip()]
        
        # Combine runtime preamble with all definitions/statements
        code = RUNTIME_PREAMBLE + "\n\n" + "\n\n".join(items)

        return code
    
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
# Define response model for {name}
{name}_response_model = {ret}

def {name}({arg_str}):
    global {name}_agent
    {name}_agent = EnsoAgent(name="{name}", instruction={instr}, model={model})
    # Filter variables to only send arguments, not the agent itself
    inputs = {{k:v for k,v in locals().items() if k != '{name}_agent' and not k.startswith('__')}}
    return {name}_agent.run(str(inputs), {name}_response_model)
"""

    def ai_body(self, args):
        d = {}
        for x in args: 
            if isinstance(x, dict): d.update(x)
        return d

    def regular_fn_def(self, args):
        # args: [name, arg_list, type_expr, stmt_block] OR [name, type_expr, stmt_block]
        if len(args) == 4:
            name, arg_str, ret_type, body = args
        else:
            name, ret_type, body, arg_str = args[0], args[1], args[2], ""
        
        # Indent each line of body by 4 spaces
        indented_body = "\n    ".join(body.split("\n"))
        return f"def {name}({arg_str}):\n    {indented_body}"

    def stmt_block(self, args):
        if not args: return "pass"
        # Don't add indentation here; let the parent rule handle it
        return "\n".join(args)
    
    def statement(self, args):
        # Unwrap the wrapper node to get the actual python string
        return args[0]

    def expr(self, args):
        # Join parts of an expression into a single Python expression string
        return " ".join(args)

    def term(self, args):
        return args[0] if args else ""

    def atom(self, args):
        return args[0]

    def let_stmt(self, args): return f"{args[0]} = {args[1]}"
    def assign_stmt(self, args): return f"{args[0]} = {args[1]}"
    def print_stmt(self, args): return f"print({args[0]})"

    def if_stmt(self, args):
        cond, body_true = args[0], args[1]
        # Indent each line of body_true by 4 spaces
        indented_body = "\n    ".join(body_true.split("\n"))
        res = f"if {cond}:\n    {indented_body}"
        if len(args) > 2: 
            indented_else = "\n    ".join(args[2].split("\n"))
            res += f"\n    else:\n    {indented_else}"
        return res

    def for_stmt(self, args):
        var_name, iterator, body = args[0], args[1], args[2]
        # Indent each line of body by 4 spaces
        indented_body = "\n    ".join(body.split("\n"))
        return f"for {var_name} in {iterator}:\n    {indented_body}"
    
    def concurrent_stmt(self, args):
        # concurrent FUNCTION(ARGS) for VAR in ITERATOR
        # args: [call_expr, var_name, iterator_expr]
        call_expr = args[0]  # e.g., "analyze(text)"
        var_name = args[1]   # e.g., "text"
        iterator = args[2]   # e.g., "texts"
        
        # Extract function name from call_expr (e.g., "analyze" from "analyze(text)")
        func_name = call_expr.split("(")[0]
        
        # Generate code that uses agent.run_concurrent()
        # First initialize the agent by calling the function with first item to set up agent
        # Then call run_concurrent with the full list
        return f"""# Initialize agent by calling function once
if not '{func_name}_agent' in globals():
    temp_init = {func_name}(next(iter({iterator})))
{func_name}_results = {func_name}_agent.run_concurrent({iterator}, '{func_name}', {func_name}_response_model)
"""
    
    def return_stmt(self, args):
        return f"return {args[0]}"
    
    def match_stmt(self, args):
        # Translate match expressions to Python if/elif chains.
        expr = args[0]
        arms = args[1:]
        
        # Build if/elif chain
        lines = []
        for i, arm in enumerate(arms):
            pattern_type, var_name, body = arm
            
            if pattern_type == 'Ok':
                condition = f"isinstance({expr}, Ok)"
                setup = f"{var_name} = {expr}.value"
            else:  # Err
                condition = f"isinstance({expr}, Err)"
                setup = f"{var_name} = {expr}.error"
            
            keyword = "if" if i == 0 else "elif"
            lines.append(f"{keyword} {condition}:")
            lines.append(f"    {setup}")
            
            # Indent each line of the body by 4 spaces
            for body_line in body.split("\n"):
                if body_line.strip():  # Only indent non-empty lines
                    lines.append(f"    {body_line}")
        
        return "\n".join(lines)
    
    def match_arm(self, args):
        # pattern_type (from ok_pattern or err_pattern), body
        pattern_type = args[0]  # This will be ('Ok', var) or ('Err', var) from the pattern handlers
        body = args[1]
        
        # Extract the type and variable name from the tuple returned by pattern handler
        if isinstance(pattern_type, tuple):
            pattern_kind, var_name = pattern_type
            return (pattern_kind, var_name, body)
        else:
            # Fallback - shouldn't happen
            return (pattern_type, 'x', body)
    
    def ok_pattern(self, args):
        # args[0] is the NAME token for the variable
        return ('Ok', args[0])
    
    def err_pattern(self, args):
        # args[0] is the NAME token for the variable
        return ('Err', args[0])
    
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
        # Indent each line of body by 4 spaces
        indented_body = "\n    ".join(body.split("\n"))
        return f"def {prefix}{slug}():\n    pass\n    {indented_body}"

    def test_body(self, args): return "\n".join(args)
    def call_expr(self, args):
        name = args[0]
        params = args[1] if len(args) > 1 else ""
        return f"{name}({params})"
    
    def list_literal(self, args):
        return f"[{', '.join(args)}]"
    
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

    def result_type(self, args):
        """Result<T, E> becomes Union[Ok[T], Err[E]] in Python"""
        ok_type = args[0]
        err_type = args[1]
        return f"Union[Ok, Err]"  # Simplified: just return Union type hint

    def struct_init(self, args): return f"{args[0]}({', '.join(args[1:])})"
    def field_init(self, args): return f"{args[0]}={args[1]}"
    def binary_expr(self, args): return f"{args[0]} {args[1]} {args[2]}"
    def NAME(self, t): return str(t)
    def TYPE_NAME(self, t): return str(t)
    def STRING(self, t): return str(t)
    def NUMBER(self, t): return str(t)
    def CONFIG_KEY(self, t): return str(t)
    def OPERATOR(self, t): return str(t)
    # We return an empty string because the Bundler has already
    # pasted the code. If we didn't use a bundler, we would generate
    # Python 'import' statements here.
    def import_def(self, args):
        return ""

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
    def regular_fn_def(self, args): return None
    def test_def(self, args): return None
    def statement(self, args): return None
    def print_stmt(self, args): return None
    def let_stmt(self, args): return None
    def assign_stmt(self, args): return None
    def for_stmt(self, args): return None
    def arg_list(self, args): return ", ".join(args)
    def arg_def(self, args): return f"{args[0]}: {args[1]}"
    def type_expr(self, args): return "Any"
    def NAME(self, t): return str(t)
    def TYPE_NAME(self, t): return str(t)
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

def compile_source(file_path):
    # STEP 1: BUNDLE IMPORTS
    full_source = bundle(file_path)
    # STEP 2: PARSE BUNDLED CODE
    # Use Earley parser instead of LALR to handle grammar ambiguities
    # (e.g., distinguishing between "expr { field: value }" in struct init
    # vs "expr { statements }" in control flow)
    parser = Lark(enso_grammar, parser='earley')
    tree = parser.parse(full_source)
    # Apply transformer to the parse tree
    transformer = EnsoTransformer()
    return transformer.transform(tree)

def analyze_source(source_code):
    parser = Lark(enso_grammar, parser='earley')
    tree = parser.parse(source_code)
    transformer = SchemaExtractor()
    try: return transformer.transform(tree)
    except Exception: return []