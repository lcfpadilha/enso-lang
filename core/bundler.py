"""
Bundler for Ensō imports.

Recursively resolves 'import "lib.enso";' directives by inlining file contents,
similar to a C preprocessor.
"""
import os
import re


def bundle(file_path, visited=None):
    """
    Recursively replaces 'import "lib.enso";' with the content of that file.
    Acts like a C pre-processor.
    
    Args:
        file_path: Path to the .enso file to bundle
        visited: Set of already-visited paths (for cycle detection)
    
    Returns:
        Bundled source code with all imports inlined
    
    Raises:
        FileNotFoundError: If an imported file doesn't exist
    """
    if visited is None:
        visited = set()
    
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
