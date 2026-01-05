"""
Error handling utilities for the Ensō compiler.
"""
import re


class EnsoCompileError(Exception):
    """Custom exception for Ensō compilation errors with line numbers and hints."""
    def __init__(self, message, line_number=None, column=None, context=None, suggestion=None):
        self.message = message
        self.line_number = line_number
        self.column = column
        self.context = context  # The offending line
        self.suggestion = suggestion  # How to fix it
        super().__init__(self._format_error())
    
    def _format_error(self):
        """Format the error message with context and suggestion."""
        lines = ["\n❌ Compilation Error"]
        if self.line_number:
            lines.append(f" at line {self.line_number}")
            if self.column:
                lines.append(f", column {self.column}")
        lines.append(":\n")
        
        # Add the error message
        lines.append(f"   {self.message}\n")
        
        # Add context if available
        if self.context:
            lines.append(f"   > {self.context}\n")
        
        # Add suggestion if available
        if self.suggestion:
            lines.append(f"   💡 {self.suggestion}\n")
        
        return "".join(lines)


def get_line_context(source_code, line_number):
    """Extract the line of code from source by line number (1-based)."""
    if not source_code or line_number is None:
        return None
    source_lines = source_code.split('\n')
    if 0 < line_number <= len(source_lines):
        return source_lines[line_number - 1].strip()
    return None


def process_escape_sequences(string_token):
    """Process escape sequences in a string token for Python output.
    
    This function takes an Ensō string with escape sequences and converts it
    to a Python-valid string literal that preserves those escape sequences.
    
    Input (from Ensō parser): '"hello\\nworld"' (with quotes, \\n is two chars)
    Output (for Python code): '"hello\\nworld"' (Python will interpret \\n as newline)
    
    The key insight: We want the Python source code to contain \\n so that when
    Python parses it, it becomes a newline. We do NOT process escapes ourselves.
    """
    # Just return the token as-is - the quotes and escapes are already correct
    # for Python to parse properly when written to a Python file
    return str(string_token)


def detect_common_error_patterns(source_code, error_msg):
    """Detect common mistakes and return helpful suggestions."""
    # Missing return type arrow
    if "fn " in source_code and "->" not in source_code:
        if re.search(r'fn\s+\w+\s*\([^)]*\)\s*{', source_code):
            return "Functions need a return type: use '-> Type' before '{'", "fn_missing_return_type"
    
    # Missing instruction in ai fn
    if "ai fn" in source_code and "instruction:" not in source_code:
        return "AI functions need an instruction: use 'instruction: \"your prompt\"'", "ai_missing_instruction"
    
    # Missing model in ai fn
    if "ai fn" in source_code and "model:" not in source_code:
        return "AI functions need a model: use 'model: \"gpt-4o\"'", "ai_missing_model"
    
    # Unmatched braces
    open_braces = source_code.count('{')
    close_braces = source_code.count('}')
    if open_braces != close_braces:
        diff = open_braces - close_braces
        brace = '{'
        return f"Unmatched braces: found {open_braces} '{brace}' but {close_braces} " + "'{}'", "unmatched_braces"
    
    # Unmatched parentheses
    open_parens = source_code.count('(')
    close_parens = source_code.count(')')
    if open_parens != close_parens:
        return f"Unmatched parentheses: found {open_parens} '(' but {close_parens} ')'", "unmatched_parens"
    
    # Missing semicolons after statements
    if re.search(r'(let|print|return|assert)\s+.+[^;}\n]\s*\n', source_code):
        return "Statements should end with ';'", "missing_semicolon"
    
    return None, None


def validate_break_continue(source_code):
    """Validate that break/continue only appear inside loops (for, while, etc)."""
    lines = source_code.split('\n')
    
    # Track which brace levels correspond to for loops
    for_loop_stack = []  # Stack of brace depths that are for loop bodies
    current_brace_depth = 0
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('//') or stripped.startswith('#'):
            continue
        
        # Track if this line contains a 'for' statement
        has_for = bool(re.search(r'\bfor\s+\w+\s+in\b', stripped))
        
        # Process braces in this line
        for i, char in enumerate(stripped):
            if char == '{':
                if has_for:
                    # This opening brace starts a for loop body
                    for_loop_stack.append(current_brace_depth)
                current_brace_depth += 1
            elif char == '}':
                current_brace_depth -= 1
                if current_brace_depth < 0:
                    current_brace_depth = 0
                # Check if we're closing a for loop
                if for_loop_stack and for_loop_stack[-1] == current_brace_depth:
                    for_loop_stack.pop()
        
        # Check for break/continue on this line
        if re.search(r'\b(break|continue)\s*;', stripped):
            # We're at break/continue if current_brace_depth is inside any for loop
            in_for_loop = len(for_loop_stack) > 0
            
            if not in_for_loop:
                keyword = "break" if "break" in stripped else "continue"
                suggestion = f"'{keyword}' can only be used inside a 'for' loop"
                raise EnsoCompileError(
                    f"'{keyword}' statement outside of loop",
                    line_number=line_num,
                    context=stripped,
                    suggestion=suggestion
                )
    
    return True
