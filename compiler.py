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
from core.transformer import EnsoTransformer
from core.introspection import SchemaExtractor

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

# ==========================================
# COMPILE AND ANALYZE FUNCTIONS
# ==========================================
# Note: EnsoTransformer and SchemaExtractor are now in core/transformer.py
# and core/introspection.py respectively. The runtime preamble is generated
# by core/runtime/get_preamble().

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