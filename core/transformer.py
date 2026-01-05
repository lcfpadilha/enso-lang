"""
Ensō AST Transformer - Converts parsed AST to Python code.

This module contains the EnsoTransformer class that transforms Lark parse trees
into executable Python code with runtime support.
"""

import re
from lark import Transformer

from core.errors import process_escape_sequences
from core.runtime import get_preamble


class EnsoTransformer(Transformer):
    """
    Transforms Ensō AST nodes into Python code strings.
    
    This transformer takes a Lark parse tree (from parsing .enso source code)
    and produces executable Python code that includes the Ensō runtime.
    """
    
    def __init__(self, runtime_preamble=None):
        """
        Initialize the transformer.
        
        Args:
            runtime_preamble: Optional custom runtime preamble. If None, uses get_preamble().
        """
        super().__init__()
        self._runtime_preamble = runtime_preamble if runtime_preamble is not None else get_preamble()
    
    def start(self, items):
        """Combine runtime preamble with all definitions/statements."""
        # Filter out empty strings from bundled imports
        items = [i for i in items if i and i.strip()]
        
        # Combine runtime preamble with all definitions/statements
        code = self._runtime_preamble + "\n\n" + "\n\n".join(items)

        return code
    
    def struct_def(self, args):
        """Transform struct definition to Pydantic BaseModel class."""
        name, *fields = args
        return f"class {name}(BaseModel):\n    " + "\n    ".join(fields)
    
    def field_def(self, args):
        """Transform field definition."""
        return f"{args[0]}: {args[1]}"

    def ai_fn_def(self, args):
        """Transform AI function definition to Python function with EnsoAgent."""
        if len(args) == 4: 
            arg_str, ret, body = args[1], args[2], args[3]
        else: 
            arg_str, ret, body = "", args[1], args[2]
        
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
        """Collect AI function body configuration."""
        d = {}
        for x in args: 
            if isinstance(x, dict): d.update(x)
        return d
    
    def system_instruction_stmt(self, args):
        """Transform system instruction statement."""
        return {"system_instruction": args[0]}
    
    def examples_block(self, args):
        """Transform examples block."""
        return {"examples": args}
    
    def example_item(self, args):
        """Transform single example item."""
        # args is a list of example_field dicts
        result = {}
        for field_dict in args:
            result.update(field_dict)
        return result
    
    def example_field(self, args):
        """Transform example field."""
        # args[0] is field name, args[1] is expr value
        field_name = args[0]
        field_value = args[1]
        return {field_name: field_value}

    def regular_fn_def(self, args):
        """Transform regular function definition to Python function."""
        # args: [name, arg_list, type_expr, stmt_block] OR [name, type_expr, stmt_block]
        if len(args) == 4:
            name, arg_str, ret_type, body = args
        else:
            name, ret_type, body, arg_str = args[0], args[1], args[2], ""
        
        # Indent each line of body by 4 spaces
        indented_body = "\n    ".join(body.split("\n"))
        return f"def {name}({arg_str}):\n    {indented_body}"

    def stmt_block(self, args):
        """Transform statement block."""
        if not args: return "pass"
        # Don't add indentation here; let the parent rule handle it
        return "\n".join(args)
    
    def statement(self, args):
        """Unwrap statement node."""
        return args[0]

    def expr(self, args):
        """Transform expression with string concatenation handling."""
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
        """Transform term."""
        return args[0] if args else ""

    def atom(self, args):
        """Transform atom."""
        return args[0]

    def let_stmt(self, args):
        """Transform let statement to Python assignment."""
        return f"{args[0]} = {args[1]}"
    
    def assign_stmt(self, args):
        """Transform assignment statement."""
        return f"{args[0]} = {args[1]}"
    
    def print_stmt(self, args):
        """Transform print statement."""
        return f"print({args[0]})"

    def if_stmt(self, args):
        """Transform if/else statement."""
        cond, body_true = args[0], args[1]
        # Indent each line of body_true by 4 spaces
        indented_body = "\n    ".join(body_true.split("\n"))
        res = f"if {cond}:\n    {indented_body}"
        if len(args) > 2: 
            indented_else = "\n    ".join(args[2].split("\n"))
            res += f"\nelse:\n    {indented_else}"
        return res

    def for_stmt(self, args):
        """Transform for loop statement."""
        var_name, iterator, body = args[0], args[1], args[2]
        # Indent each line of body by 4 spaces
        indented_body = "\n    ".join(body.split("\n"))
        return f"for {var_name} in {iterator}:\n    {indented_body}"
    
    def concurrent_stmt(self, args):
        """Transform concurrent statement for parallel AI calls."""
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
        """Transform return statement."""
        return f"return {args[0]}"
    
    def break_stmt(self, args):
        """Transform break statement."""
        return "break"
    
    def continue_stmt(self, args):
        """Transform continue statement."""
        return "continue"
    
    def match_stmt(self, args):
        """Transform match expression to Python if/elif chain."""
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
        """Transform match arm."""
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
        """Transform Ok pattern."""
        return ('Ok', args[0])
    
    def err_pattern(self, args):
        """Transform Err pattern."""
        return ('Err', args[0])
    
    def test_def(self, args):
        """Transform test definition to Python test function."""
        # args: [test_type | None, STRING, test_body]
        # test_type is optional, so args[0] is None for regular tests
        # and a Tree for "ai" tests
        if args[0] is not None:
            # AI test (has test_type)
            is_ai = True
            name = args[1].replace('"', '')
            body = args[2]
        else:
            # Regular test (test_type is None)
            is_ai = False
            name = args[1].replace('"', '')
            body = args[2]
        slug = name.replace(" ", "_")
        prefix = "test_AI_" if is_ai else "test_"
        # Indent each line of body by 4 spaces
        indented_body = "\n    ".join(body.split("\n"))
        return f"def {prefix}{slug}():\n    pass\n    {indented_body}"

    def test_body(self, args):
        """Transform test body."""
        return "\n".join(args)
    
    def call_expr(self, args):
        """Transform function call expression."""
        name = args[0]
        params = args[1] if len(args) > 1 else ""
        return f"{name}({params})"
    
    def list_literal(self, args):
        """Transform list literal."""
        return f"[{', '.join(args)}]"
    
    def args_call(self, args):
        """Transform call arguments."""
        return ", ".join(args)
    
    def mock_stmt(self, args):
        """Transform mock statement."""
        return f"MOCKS['{args[0]}'] = {args[1]}"
    
    def assertion(self, args):
        """Transform assertion."""
        return f"assert {args[0]}"
    
    def prop_access(self, args):
        """Transform property access."""
        return ".".join(args)

    def instruction_stmt(self, args):
        """Transform instruction statement."""
        return {"instruction": args[0]}
    
    def config_stmt(self, args):
        """Transform config statement."""
        return {args[0]: args[1]}
    
    def arg_list(self, args):
        """Transform argument list."""
        return ", ".join(args)
    
    def arg_def(self, args):
        """Transform argument definition."""
        return f"{args[0]}: {args[1]}"
    
    def type_expr(self, args):
        """Transform type expression."""
        return args[0]  # Pass-through for the one chosen child

    def simple_type(self, args):
        """Transform simple type (String, Int, Float) to Python type."""
        arg0 = str(args[0])
        m = {"String": "str", "Int": "int", "Float": "float"}
        return m.get(arg0, arg0)

    def list_type(self, args):
        """Transform List<T> to Python List[T]."""
        return f"List[{args[0]}]"

    def enum_type(self, args):
        """Transform Enum<...> to Python Literal[...]."""
        options = ", ".join([str(x) for x in args])
        return f"Literal[{options}]"

    def result_type(self, args):
        """Transform Result<T, E> to response model marker."""
        ok_type = args[0]
        err_type = args[1]
        # Store the response type for AI functions (the T in Result<T, E>)
        # Return a special marker that ai_fn_def can parse
        return f"__RESULT_TYPE__({ok_type})__"

    def struct_init(self, args):
        """Transform struct initialization."""
        return f"{args[0]}({', '.join(args[1:])})"
    
    def field_init(self, args):
        """Transform field initialization."""
        return f"{args[0]}={args[1]}"
    
    def binary_expr(self, args):
        """Transform binary expression with string concatenation handling."""
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
    
    def NAME(self, t):
        """Transform NAME token."""
        return str(t)
    
    def TYPE_NAME(self, t):
        """Transform TYPE_NAME token."""
        return str(t)
    
    def STRING(self, t):
        """Transform STRING token with escape sequence processing."""
        return process_escape_sequences(str(t))
    
    def STRING_WITH_VARS(self, t):
        """Transform STRING_WITH_VARS token with escape sequence processing."""
        return process_escape_sequences(str(t))
    
    def NUMBER(self, t):
        """Transform NUMBER token."""
        return str(t)
    
    def CONFIG_KEY(self, t):
        """Transform CONFIG_KEY token."""
        return str(t)
    
    def OPERATOR(self, t):
        """Transform OPERATOR token."""
        return str(t)
    
    def import_def(self, args):
        """Transform import definition (no-op, bundler handles imports)."""
        # We return an empty string because the Bundler has already
        # pasted the code. If we didn't use a bundler, we would generate
        # Python 'import' statements here.
        return ""
