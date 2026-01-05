# Ensō Runtime Components
"""
Runtime modules that get injected into compiled Ensō code.

These are real Python files that provide IDE support and testability,
but are concatenated into a single preamble string at compile time.
"""

import os


def get_preamble():
    """
    Read and concatenate all runtime modules into a single preamble string.
    
    This is injected at the top of every compiled .enso file to make
    the output self-contained (no external dependencies on enso_lang).
    """
    runtime_dir = os.path.dirname(__file__)
    
    # Order matters - dependencies must come first
    modules = [
        'preamble_header.py',  # Imports and globals
        'result.py',           # Ok, Err, Result, AIError
        'config.py',           # load_model_config, resolve_refs
        'drivers.py',          # LLMDriver, OpenAIDriver, GeminiDriver, LocalDriver
        'agent.py',            # EnsoAgent, Probabilistic
        'test_runner.py',      # run_tests
    ]
    
    parts = []
    for module in modules:
        path = os.path.join(runtime_dir, module)
        with open(path, 'r') as f:
            content = f.read()
            # Skip module docstrings and imports (they're in preamble_header)
            if module != 'preamble_header.py':
                # Remove the "if __name__" block if present (for testing)
                lines = content.split('\n')
                filtered = []
                skip_main = False
                for line in lines:
                    if line.startswith('if __name__'):
                        skip_main = True
                    elif skip_main and line and not line.startswith(' ') and not line.startswith('\t'):
                        skip_main = False
                    if not skip_main:
                        filtered.append(line)
                content = '\n'.join(filtered)
            parts.append(content)
    
    return '\n\n'.join(parts)
