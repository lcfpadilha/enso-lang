"""
Ensō Grammar Definition.

This module contains the Lark grammar for the Ensō DSL.
"""

enso_grammar = r"""
    start: (import_def | struct_def | ai_fn_def | regular_fn_def | test_def | statement)*

    // --- Imports ---
    import_def: "import" STRING ";"

    // --- Definitions ---
    struct_def: "struct" NAME "{" field_def* "}"
    field_def: NAME ":" type_expr [","]

    // AI Function (Declarative)
    ai_fn_def: "ai" "fn" NAME "(" arg_list? ")" "->" type_expr "{" ai_body "}"
    ai_body: system_instruction_stmt? instruction_stmt (config_stmt | examples_block)* validate_block?

    // Regular Function (Imperative)
    regular_fn_def: "fn" NAME "(" arg_list? ")" ["->" type_expr] "{" stmt_block "}"
    stmt_block: statement*

    // --- Testing ---
    test_def: "test" [test_type] STRING "{" test_body "}"
    test_type: "ai"
    test_body: (mock_stmt | statement)*
    
    mock_stmt: "mock" NAME "=>" expr ";"

    // --- Details ---
    system_instruction_stmt: "system_instruction" ":" STRING_WITH_VARS [","]
    instruction_stmt: "instruction" ":" STRING_WITH_VARS [","]
    examples_block: "examples" ":" "[" example_item* "]" [","]
    example_item: "(" example_field ("," example_field)* ")" [","]
    example_field: NAME ":" expr
    config_stmt: CONFIG_KEY ":" (NUMBER | STRING | expr) [","]
    CONFIG_KEY: "temperature" | "model" | "max_tokens" | "top_p" | "top_k"
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
    statement: let_stmt | assign_stmt | print_stmt | return_stmt | if_stmt | for_stmt | match_stmt | concurrent_stmt | assertion | break_stmt | continue_stmt | expr_stmt

    let_stmt: "let" NAME "=" expr ";"
    concurrent_stmt: "concurrent" call_expr "for" NAME "in" expr ";"
    assign_stmt: NAME "=" expr ";"
    print_stmt: "print" "(" expr ")" ";"
    return_stmt: "return" expr ";"
    break_stmt: "break" ";"
    continue_stmt: "continue" ";"
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
    STRING_WITH_VARS: /"(?:[^"\\]|\\.|{[a-zA-Z_]\w*})*"/
    STRING: /"(?:[^"\\]|\\.)*"/
    NUMBER: /\d+(\.\d+)?/
    NAME: /(?!(?:let|print|return|if|else|for|in|import|struct|ai|fn|test|mock|assert|instruction|temperature|model|validate|List|Enum|Result|match|Ok|Err|concurrent|system_instruction|examples|break|continue)\b)[a-zA-Z_]\w*/
    
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
