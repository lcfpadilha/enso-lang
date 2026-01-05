# Ensō Language - Core Compiler Components
"""
Core modules for the Ensō compiler:
- errors: Error handling and validation utilities
- bundler: Import resolution (inlines imported files)
- grammar: Lark grammar definition for the Ensō DSL
- runtime: Runtime preamble generation for compiled output
- transformer: AST to Python code transformation
- introspection: Schema extraction for AI functions
"""

from .errors import EnsoCompileError
from .bundler import bundle
from .grammar import enso_grammar
from .transformer import EnsoTransformer
from .introspection import SchemaExtractor

__all__ = [
    'EnsoCompileError',
    'bundle',
    'enso_grammar',
    'EnsoTransformer',
    'SchemaExtractor',
]
