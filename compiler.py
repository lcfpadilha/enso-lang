import sys
import os
import re
from lark import Lark, Transformer

# Import from the core package
from core.errors import (
    EnsoCompileError,
    get_line_context,
    process_escape_sequences,
    detect_common_error_patterns,
    validate_break_continue,
)
from core.bundler import bundle
from core.grammar import enso_grammar
from core.runtime import get_preamble

# Global verbose flag
_VERBOSE = False

def set_verbose(value):
    """Set the global verbose flag."""
    global _VERBOSE
    _VERBOSE = value

def debug_log(message):
    """Log a debug message to stderr if verbose mode is enabled."""
    if _VERBOSE:
        print(f"\033[94mDEBUG:\033[0m {message}", file=sys.stderr)

RUNTIME_PREAMBLE = get_preamble()

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
        
        name = args[0]
        instr = body['instruction']
        model = body.get('model', '"gpt-4o"')
        system_instr = body.get('system_instruction', None)
        examples = body.get('examples', None)
        temperature = body.get('temperature', None)
        
        # Extract parameter names from arg_str
        param_names = []
        if arg_str:
            for param in arg_str.split(','):
                param_name = param.split(':')[0].strip()
                if param_name:
                    param_names.append(param_name)
        
        # Validate interpolation variables
        import re
        instr_str = instr.strip('"\'f')
        vars_in_instr = re.findall(r'{([a-zA-Z_]\w*)}', instr_str)
        undefined_vars = set(vars_in_instr) - set(param_names)
        if undefined_vars:
            raise ValueError(f"Undefined variables in instruction: {undefined_vars}")
        
        # Generate examples hoisting code
        examples_code = ""
        if examples:
            examples_list = []
            for ex in examples:
                examples_list.append(ex)
            examples_code = f"\n{name}_examples = {examples_list}\n"
        
        # Convert instruction string to f-string if it has variables
        if vars_in_instr:
            instr = f'f{instr}'
        
        # Extract response model from Result<T, E> return type
        response_model = ret
        if "__RESULT_TYPE__(" in ret:
            # Extract the inner type from __RESULT_TYPE__(SomeType)__
            match = re.search(r'__RESULT_TYPE__\(([^)]+)\)__', ret)
            if match:
                response_model = match.group(1)
        
        # Build EnsoAgent arguments
        agent_args = [f'name="{name}"', f'instruction={instr}', f'model={model}']
        if system_instr:
            agent_args.insert(1, f'system_instruction={system_instr}')
        if examples:
            agent_args.append(f'examples={name}_examples')
        if temperature:
            agent_args.append(f'temperature={temperature}')
        
        agent_args_str = ", ".join(agent_args)
        
        return f"""{examples_code}
# Define response model for {name}
{name}_response_model = {response_model}

def {name}({arg_str}):
    global {name}_agent
    {name}_agent = EnsoAgent({agent_args_str})
    # Filter variables to only send arguments, not the agent itself
    inputs = {{k:v for k,v in locals().items() if k != '{name}_agent' and not k.startswith('__')}}
    return {name}_agent.run(str(inputs), {name}_response_model)
"""

    def ai_body(self, args):
        d = {}
        for x in args: 
            if isinstance(x, dict): d.update(x)
        return d
    
    def system_instruction_stmt(self, args):
        return {"system_instruction": args[0]}
    
    def examples_block(self, args):
        return {"examples": args}
    
    def example_item(self, args):
        # args is a list of example_field dicts
        result = {}
        for field_dict in args:
            result.update(field_dict)
        return result
    
    def example_field(self, args):
        # args[0] is field name, args[1] is expr value
        field_name = args[0]
        field_value = args[1]
        return {field_name: field_value}

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
        # Handle string concatenation with automatic type conversion
        result = " ".join(args)
        
        # Post-process for string concatenation with type coercion
        # If we have something like: "string" + variable
        # Convert to: "string" + str(variable)
        # BUT: Don't wrap:
        #   - property access (e.g., p.name or candidate.name) - followed by dot
        #   - function calls (e.g., get_value()) - followed by paren
        # Use (?![\w.(\[]）to prevent matching if followed by word char, dot, paren, or bracket
        result = re.sub(r'("(?:[^"\\]|\\.)*")\s*\+\s*(?!str\()(\w+)(?![\w.(\[])', r'\1 + str(\2)', result)
        
        return result

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
            res += f"\nelse:\n    {indented_else}"
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
    
    def break_stmt(self, args):
        return "break"
    
    def continue_stmt(self, args):
        return "continue"
    
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
        """Result<T, E> -> extract T for response model, return type hint is Union[Ok, Err]"""
        ok_type = args[0]
        err_type = args[1]
        # Store the response type for AI functions (the T in Result<T, E>)
        # Return a special marker that ai_fn_def can parse
        return f"__RESULT_TYPE__({ok_type})__"

    def struct_init(self, args): return f"{args[0]}({', '.join(args[1:])})"
    def field_init(self, args): return f"{args[0]}={args[1]}"
    def binary_expr(self, args):
        # Handle string concatenation with automatic type conversion
        # If operator is '+' and either operand is a string, auto-convert the other
        op = args[1].strip()
        if op == '+':
            left = str(args[0])
            right = str(args[2])
            # Check if either side is a string literal (contains quotes)
            is_left_string = '"' in left
            is_right_string = '"' in right
            
            # If one side is a string and the other isn't, wrap non-string with str()
            # BUT: Don't wrap property access (contains dots) - these already return values
            # e.g., "text" + obj.prop should be "text" + obj.prop, not "text" + str(obj).prop
            if is_left_string and not is_right_string:
                # Only wrap if right doesn't contain a dot (property access)
                if '.' not in right:
                    right = f"str({right})"
            elif is_right_string and not is_left_string:
                # Only wrap if left doesn't contain a dot (property access)
                if '.' not in left:
                    left = f"str({left})"
            
            return f"{left} {op} {right}"
        return f"{args[0]} {args[1]} {args[2]}"
    def NAME(self, t): return str(t)
    def TYPE_NAME(self, t): return str(t)
    def STRING(self, t): return process_escape_sequences(str(t))
    def STRING_WITH_VARS(self, t): return process_escape_sequences(str(t))
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
    def system_instruction_stmt(self, t): return None
    def examples_block(self, t): return None
    def example_item(self, t): return None
    def example_field(self, t): return None
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

def compile_source(file_path, source_code=None):
    # STEP 1: BUNDLE IMPORTS
    if source_code is None:
        full_source = bundle(file_path)
    else:
        # Use provided source code directly (for stdin)
        full_source = source_code
    
    debug_log(f"Compiling source: {file_path}")
    
    # STEP 1.5: VALIDATE BREAK/CONTINUE
    try:
        validate_break_continue(full_source)
    except EnsoCompileError:
        raise  # Re-raise our custom errors
    
    # STEP 2: PARSE BUNDLED CODE
    # Use Earley parser instead of LALR to handle grammar ambiguities
    try:
        parser = Lark(enso_grammar, parser='earley')
        tree = parser.parse(full_source)
    except Exception as e:
        # Extract error information from Lark exception
        error_msg = str(e)
        line_number = None
        column = None
        
        # Try to extract line/column from error message
        # Lark typically reports: "Error trying to process rule \"...\": ... expected ... at line X column Y"
        match = re.search(r'line (\d+) col (\d+)', error_msg)
        if match:
            line_number = int(match.group(1))
            column = int(match.group(2))
        
        # Get line context
        context = get_line_context(full_source, line_number) if line_number else None
        
        # Try to detect common patterns
        suggestion_text, error_type = detect_common_error_patterns(full_source, error_msg)
        
        # If no specific pattern matched, provide generic hint
        if not suggestion_text:
            suggestion_text = "Check syntax around this line"
        
        raise EnsoCompileError(
            message="Syntax error",
            line_number=line_number,
            column=column,
            context=context,
            suggestion=suggestion_text
        )
    
    # STEP 3: TRANSFORM
    try:
        transformer = EnsoTransformer()
        return transformer.transform(tree)
    except EnsoCompileError:
        raise  # Re-raise our custom errors
    except ValueError as e:
        # Handle semantic errors (e.g., undefined variables in prompts)
        error_msg = str(e)
        raise EnsoCompileError(
            message=error_msg,
            suggestion="Check variable names and function parameters"
        )
    except Exception as e:
        # Catch any other transformation errors
        raise EnsoCompileError(
            message=f"Transformation error: {str(e)}",
            suggestion="Check syntax and types"
        )

def analyze_source(source_code):
    parser = Lark(enso_grammar, parser='earley')
    tree = parser.parse(source_code)
    transformer = SchemaExtractor()
    try: return transformer.transform(tree)
    except Exception: return []