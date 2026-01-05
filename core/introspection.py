"""
Ensō Schema Extraction - Introspection for AI function signatures.

This module contains the SchemaExtractor class that extracts function
signatures and metadata from parsed Ensō code for SDK generation and analysis.
"""

from lark import Transformer


class SchemaExtractor(Transformer):
    """
    Extracts schema information from Ensō AST for introspection.
    
    This transformer analyzes parsed Ensō code and extracts metadata about
    AI functions (name, arguments, return type) for use in SDK generation,
    documentation, and analysis tools.
    
    Unlike EnsoTransformer which generates executable code, SchemaExtractor
    only extracts structural information and ignores implementation details.
    """
    
    def start(self, items):
        """Extract only function definitions from the AST."""
        return [i for i in items if isinstance(i, dict) and i.get('type') == 'function']
    
    def ai_fn_def(self, args):
        """Extract AI function signature."""
        if len(args) == 4:
            arg_str, ret, body = args[1], args[2], args[3]
        else:
            arg_str, ret, body = "", args[1], args[2]
        
        arg_list = []
        if arg_str:
            for part in arg_str.split(","):
                p = part.split(":")
                arg_list.append({"name": p[0].strip(), "type": p[1].strip()})
        
        return {
            "type": "function",
            "name": args[0],
            "args": arg_list,
            "return": ret
        }
    
    # Ignored nodes - return None for non-function definitions
    def struct_def(self, args):
        """Ignore struct definitions."""
        return None
    
    def regular_fn_def(self, args):
        """Ignore regular function definitions."""
        return None
    
    def test_def(self, args):
        """Ignore test definitions."""
        return None
    
    def statement(self, args):
        """Ignore statements."""
        return None
    
    def print_stmt(self, args):
        """Ignore print statements."""
        return None
    
    def let_stmt(self, args):
        """Ignore let statements."""
        return None
    
    def assign_stmt(self, args):
        """Ignore assign statements."""
        return None
    
    def for_stmt(self, args):
        """Ignore for statements."""
        return None
    
    def arg_list(self, args):
        """Transform argument list for extraction."""
        return ", ".join(args)
    
    def arg_def(self, args):
        """Transform argument definition for extraction."""
        return f"{args[0]}: {args[1]}"
    
    def type_expr(self, args):
        """Transform type expression - pass through the actual type."""
        return args[0]
    
    def simple_type(self, args):
        """Transform simple type - preserve original name for SDK generation."""
        return str(args[0])
    
    def list_type(self, args):
        """Transform List<T> type."""
        return f"List<{args[0]}>"
    
    def enum_type(self, args):
        """Transform Enum type."""
        options = ", ".join([str(x) for x in args])
        return f"Enum<{options}>"
    
    def result_type(self, args):
        """Transform Result<T, E> type."""
        return f"Result<{args[0]}, {args[1]}>"
    
    def NAME(self, t):
        """Transform NAME token."""
        return str(t)
    
    def TYPE_NAME(self, t):
        """Transform TYPE_NAME token."""
        return str(t)
    
    def STRING(self, t):
        """Transform STRING token."""
        return str(t)
    
    def NUMBER(self, t):
        """Transform NUMBER token."""
        return str(t)
    
    # Additional ignored nodes
    def ai_body(self, t):
        """Ignore AI body."""
        return None
    
    def instruction_stmt(self, t):
        """Ignore instruction statement."""
        return None
    
    def system_instruction_stmt(self, t):
        """Ignore system instruction statement."""
        return None
    
    def examples_block(self, t):
        """Ignore examples block."""
        return None
    
    def example_item(self, t):
        """Ignore example item."""
        return None
    
    def example_field(self, t):
        """Ignore example field."""
        return None
    
    def config_stmt(self, t):
        """Ignore config statement."""
        return None
    
    def validate_block(self, t):
        """Ignore validate block."""
        return None
    
    def call_expr(self, t):
        """Ignore call expression."""
        return None
    
    def args_call(self, t):
        """Ignore args call."""
        return None
    
    def struct_init(self, t):
        """Ignore struct initialization."""
        return None
    
    def field_init(self, t):
        """Ignore field initialization."""
        return None
    
    def binary_expr(self, t):
        """Ignore binary expression."""
        return None
    
    def prop_access(self, t):
        """Ignore property access."""
        return None
    
    def test_body(self, t):
        """Ignore test body."""
        return None
    
    def mock_stmt(self, t):
        """Ignore mock statement."""
        return None
    
    def assertion(self, t):
        """Ignore assertion."""
        return None
