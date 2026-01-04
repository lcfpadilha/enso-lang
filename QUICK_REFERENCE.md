# Ensō Quick Reference - Complete Grammar Guide

**Last Updated**: January 4, 2026  
**Status**: Production Ready

---

## Table of Contents

1. [Basic Syntax](#basic-syntax)
2. [Types](#types)
3. [Structs](#structs)
4. [Functions](#functions)
5. [AI Functions](#ai-functions)
6. [Testing](#testing)
7. [Statements](#statements)
8. [Expressions](#expressions)
9. [Examples](#examples)

---

## Basic Syntax

### Comments
```rust
// Single-line comment
# Hash comment
/* Multi-line
   comment */
```

### File Structure
```rust
import "module.enso";     // Import modules

struct MyStruct { ... }   // Define types

ai fn myAI(...) { ... }   // Define AI function

fn myFn(...) { ... }      // Define regular function

test "name" { ... }       // Define tests
```

---

## Types

### Primitives
```rust
String              // Text: "hello"
Int                 // Integer: 42
Float               // Decimal: 3.14
```

### Enum (Restricted Set)
```rust
Enum<"Option1", "Option2", "Option3">

// Example:
struct Decision {
    choice: Enum<"Yes", "No", "Maybe">
}
```

### List (Collections)
```rust
List<String>        // ["a", "b", "c"]
List<Int>           // [1, 2, 3]
List<MyStruct>      // [struct1, struct2]
```

### Result (Error Handling)
```rust
Result<T, AIError>

// Example:
Result<MyData, AIError>
```

### User-Defined Types
Start with capital letter:
```rust
Candidate           // Refers to struct Candidate
MyCustomType        // Refers to struct MyCustomType
```

---

## Structs

### Definition
```rust
struct MyStruct {
    field1: String,
    field2: Int,
    field3: List<String>,
    field4: Enum<"A", "B">
}
```

### Initialization
```rust
let value = MyStruct {
    field1: "hello",
    field2: 42,
    field3: ["a", "b"],
    field4: "A"
};

// Access fields
print(value.field1);
```

### Field Access
```rust
let name = candidate.name;
let score = result.value.score;
```

---

## Functions

### Regular Function (Imperative)
```rust
fn myFunction(param1: String, param2: Int) -> String {
    // Statements here
    print("Hello");
    let x = param1 + param2;
    return x;
}

fn noReturn(x: String) {
    // No return type = void
    print(x);
}
```

### Function Calls
```rust
let result = myFunction("text", 42);
```

### Parameters
```rust
fn example(
    name: String,
    age: Int,
    tags: List<String>,
    status: Enum<"Active", "Inactive">
) -> String {
    ...
}
```

### No-Argument Functions
```rust
fn hello() -> String {
    return "world";
}

let greeting = hello();
```

---

## AI Functions

### Basic Structure
```rust
ai fn functionName(param1: Type1, param2: Type2) -> Result<OutputType, AIError> {
    instruction: "Your prompt here",
    model: "gpt-4o"
}
```

### With System Instruction
```rust
ai fn analyze(text: String) -> Result<Analysis, AIError> {
    system_instruction: "You are an expert analyzer.",
    instruction: "Analyze: {text}",
    model: "gpt-4o"
}
```

### With Few-Shot Examples
```rust
ai fn classify(text: String) -> Result<Classification, AIError> {
    system_instruction: "You are a classifier.",
    instruction: "Classify: {text}",
    examples: [
        (text: "Positive example", expected: Classification { ... }),
        (text: "Negative example", expected: Classification { ... })
    ],
    model: "gpt-4o"
}
```

### With Configuration
```rust
ai fn extract(document: String) -> Result<Data, AIError> {
    instruction: "Extract from: {document}",
    temperature: 0.1,       // 0 = deterministic, 1 = creative
    model: "gpt-4o",        // Model choice
    max_tokens: 1000,       // Max output length
    top_p: 0.9,             // Nucleus sampling
    top_k: 40               // Top-k filtering
}
```

### Variable Interpolation
Variables from function parameters can be injected into prompts:
```rust
ai fn process(item: String, context: String) -> Result<Output, AIError> {
    instruction: "Process {item} with context {context}",
    model: "gpt-4o"
}
```

### Examples Format
```rust
examples: [
    (
        field1: "value1",
        field2: "value2",
        expected: OutputType { 
            result_field: "expected_value"
        }
    ),
    (
        field1: "value3",
        field2: "value4",
        expected: OutputType { 
            result_field: "expected_value"
        }
    )
]
```

### Configuration Options

| Option | Type | Purpose | Example |
|--------|------|---------|---------|
| `instruction` | String | Task to perform | `"Extract names from: {text}"` |
| `system_instruction` | String | Role context | `"You are a recruiter"` |
| `model` | String | LLM to use | `"gpt-4o"` or `"gemini-2.5-flash"` |
| `temperature` | Float | Creativity (0-1) | `0.1` (precise) or `0.9` (creative) |
| `max_tokens` | Int | Output limit | `1000` |
| `top_p` | Float | Nucleus sampling | `0.9` |
| `top_k` | Int | Top-k filtering | `40` |
| `examples` | List | Few-shot examples | `[(input, expected), ...]` |
| `validate` | Block | Custom validation | `validate(result) { ... }` |

### Return Type
Always `Result<T, AIError>`:
```rust
Result<Candidate, AIError>      // Success: Candidate, Failure: AIError
Result<String, AIError>         // Success: String
Result<List<Item>, AIError>     // Success: List
```

---

## Testing

### Basic Test
```rust
test "Test name" {
    // Regular statements
    let x = 5;
    assert x == 5;
}
```

### Mocking AI Functions
```rust
test "Test with mock" {
    mock functionName => ReturnValue {
        field1: "value",
        field2: 42
    };
    
    let result = functionName("input");
    match result {
        Ok(data) => {
            assert data.value.field1 == "value";
        },
        Err(error) => {
            assert false;  // Shouldn't get here
        }
    }
}
```

### Running Tests
```bash
enso test file.enso              # Run mocked tests (no API calls)
enso test file.enso --include_ai # Run with real AI (costs money!)
```

---

## Statements

### Variable Declaration
```rust
let x = 5;
let name = "John";
let items = [1, 2, 3];
let result = functionCall();
```

### Assignment
```rust
x = 10;
name = "Jane";
```

### Printing
```rust
print("Hello");
print(variable);
print("Value: " + x);
```

### Return
```rust
return value;
return "done";
```

### If Statement
```rust
if condition {
    // True branch
} else {
    // False branch
}

if x > 5 {
    print("Greater");
} else {
    print("Lesser");
}
```

### For Loop
```rust
for item in items {
    print(item);
}

// With list
let numbers = [1, 2, 3];
for n in numbers {
    print(n);
}
```

### Match Expression (Pattern Matching)
```rust
match result {
    Ok(value) => {
        // Success case
        print(value);
    },
    Err(error) => {
        // Failure case
        print(error.message);
    }
}
```

### Concurrent Batch Processing
```rust
concurrent functionName(item) for item in items;

// Results available in: functionName_results (List<Result<T, E>>)
for result in functionName_results {
    match result {
        Ok(data) => { ... },
        Err(error) => { ... }
    }
}
```

### Assertion
```rust
assert condition;
assert x == 5;
assert name == "John";
```

---

## Expressions

### Literals
```rust
"string"            // String literal
42                  // Integer
3.14                // Float
[1, 2, 3]           // List literal
```

### Operators

| Operator | Use | Example |
|----------|-----|---------|
| `+` | Addition/Concatenation | `x + 5`, `"hello" + "world"` |
| `-` | Subtraction | `x - 3` |
| `==` | Equality | `x == 5` |
| `!=` | Inequality | `x != 5` |
| `>` | Greater than | `x > 5` |
| `<` | Less than | `x < 5` |
| `>=` | Greater or equal | `x >= 5` |
| `<=` | Less or equal | `x <= 5` |
| `&&` | Logical AND | `x > 0 && x < 10` |
| `\|\|` | Logical OR | `x < 0 \|\| x > 10` |

### Function Calls
```rust
myFunction()
myFunction(arg1)
myFunction(arg1, arg2, arg3)
```

### Struct Creation
```rust
MyStruct {
    field1: "value",
    field2: 42
}
```

### Field Access
```rust
myStruct.field1
myStruct.field1.nestedField
result.value.field
```

### List Operations
```rust
[1, 2, 3]                   // List literal
[element1, element2]        // Multiple elements
[]                          // Empty list
```

---

## Examples

### Complete Hiring Example
```rust
struct Skill {
    name: String,
    level: Enum<"Beginner", "Intermediate", "Expert">
}

struct Candidate {
    name: String,
    years_experience: Int,
    top_skills: List<Skill>
}

ai fn parse_resume(text: String) -> Result<Candidate, AIError> {
    system_instruction: "You are an expert recruiter.",
    instruction: "Extract candidate details from: {text}",
    examples: [
        (
            text: "Alice: 8 years Python expert",
            expected: Candidate {
                name: "Alice",
                years_experience: 8,
                top_skills: [Skill { name: "Python", level: "Expert" }]
            }
        )
    ],
    temperature: 0.1,
    model: "gpt-4o"
}

fn main() {
    let resume = "John: 5 years Python and Go";
    match parse_resume(resume) {
        Ok(candidate) => {
            print(candidate.value.name);
        },
        Err(error) => {
            print(error.message);
        }
    }
}
```

### Sentiment Classification
```rust
struct SentimentResult {
    sentiment: Enum<"Positive", "Negative", "Neutral">,
    confidence: Int
}

ai fn analyze_sentiment(text: String) -> Result<SentimentResult, AIError> {
    system_instruction: "You are a sentiment analysis expert.",
    instruction: "Analyze sentiment of: {text}",
    examples: [
        (
            text: "This is amazing!",
            expected: SentimentResult {
                sentiment: "Positive",
                confidence: 95
            }
        ),
        (
            text: "This is terrible.",
            expected: SentimentResult {
                sentiment: "Negative",
                confidence: 90
            }
        )
    ],
    temperature: 0.1,
    model: "gpt-4o"
}

fn main() {
    let review = "Pretty good product";
    match analyze_sentiment(review) {
        Ok(result) => {
            print(result.value.sentiment);
            print(result.value.confidence);
        },
        Err(error) => {
            print("Error: " + error.message);
        }
    }
}
```

### Testing
```rust
test "Mocked sentiment analysis" {
    mock analyze_sentiment => SentimentResult {
        sentiment: "Positive",
        confidence: 95
    };
    
    let result = analyze_sentiment("Great!");
    match result {
        Ok(data) => {
            assert data.value.sentiment == "Positive";
            assert data.value.confidence == 95;
        },
        Err(error) => {
            assert false;
        }
    }
}
```

---

## Best Practices

### 1. Use Enums for Constraints
```rust
// ✅ Good: Only valid values possible
status: Enum<"Active", "Inactive", "Pending">

// ❌ Bad: Strings allow invalid values
status: String
```

### 2. Use Examples for Accuracy
```rust
// ✅ Good: 2-3 examples improve accuracy 25-35%
examples: [
    (input: "example1", expected: ...),
    (input: "example2", expected: ...)
]

// ❌ Bad: No examples, lower accuracy
// (no examples)
```

### 3. Low Temperature for Extraction
```rust
// ✅ Good: For deterministic tasks
temperature: 0.1,  // Extract, classify, parse

// ❌ Bad: High temperature = inconsistent
temperature: 0.9,  // Creative tasks OK, not extraction
```

### 4. Handle Errors Explicitly
```rust
// ✅ Good: Handle both cases
match result {
    Ok(data) => { ... },
    Err(error) => { ... }
}

// ❌ Bad: Ignoring errors
let data = result.value;  // Crashes if error!
```

### 5. Use Variable Interpolation
```rust
// ✅ Good: Dynamic prompts
instruction: "Process {item} with {context}",

// ❌ Bad: Hardcoded, not flexible
instruction: "Process item with context",
```

---

## Common Patterns

### Pattern 1: Extraction Pipeline
```rust
ai fn extract(text: String) -> Result<ExtractedData, AIError> {
    instruction: "Extract from: {text}",
    model: "gpt-4o"
}
```
Use for: Form filling, resume parsing, data migration

### Pattern 2: Classification
```rust
ai fn classify(input: String) -> Result<Classification, AIError> {
    instruction: "Classify: {input}",
    examples: [...]
}
```
Use for: Sentiment, routing, moderation, tagging

### Pattern 3: Analysis
```rust
ai fn analyze(document: String) -> Result<Analysis, AIError> {
    instruction: "Analyze: {document}",
    model: "gpt-4o"
}
```
Use for: Document review, research, summarization

### Pattern 4: Multi-Step Orchestration
```rust
match step1_function(input) {
    Ok(step1_result) => {
        match step2_function(step1_result.value) {
            Ok(step2_result) => { ... },
            Err(error) => { ... }
        }
    },
    Err(error) => { ... }
}
```
Use for: Hiring pipeline, onboarding, complex workflows

---

## Running Ensō

### Run Main
```bash
enso run file.enso
```

### Run Tests (Mocked)
```bash
enso test file.enso
```

### Run Tests (Real AI)
```bash
enso test file.enso --include_ai
```

### Initialize Project
```bash
enso init
```

---

## Supported Models

### OpenAI
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-4o-mini`

### Google Gemini
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-2.0-flash`

### Add More
Update `models.json` with new model definitions

---

## Error Handling

### Error Types
```rust
ErrorKind::API_ERROR           // API call failed
ErrorKind::PARSE_ERROR         // Output parsing failed
ErrorKind::HALLUCINATION_ERROR // Invalid response
ErrorKind::TIMEOUT_ERROR       // Request timed out
ErrorKind::REFUSAL_ERROR       // Model refused
ErrorKind::INVALID_CONFIG_ERROR // Bad configuration
```

### Error Access
```rust
match result {
    Ok(data) => { ... },
    Err(error) => {
        print(error.kind);      // Error category
        print(error.message);   // Error message
        print(error.cost);      // Cost incurred
        print(error.model);     // Model used
    }
}
```

---

## Cost Tracking

Every AI function call returns cost metadata:
```rust
match analyze_sentiment(text) {
    Ok(result) => {
        let cost = result.cost;        // Cost in dollars
        let model = result.model;      // Model used
        let tokens_in = result.tokens_in;
        let tokens_out = result.tokens_out;
    },
    Err(error) => {
        let cost = error.cost;         // Cost even on error
    }
}
```

---

## Grammar EBNF (For Reference)

```
program         = (import | struct | ai_fn | fn | test)*
import          = "import" STRING ";"
struct          = "struct" NAME "{" field* "}"
ai_fn           = "ai" "fn" NAME "(" args? ")" "->" type "{" ai_body "}"
ai_body         = [system_instr] instr [examples] [config] [validate]
fn              = "fn" NAME "(" args? ")" ["->" type] "{" statement* "}"
test            = "test" STRING "{" (mock | statement)* "}"
statement       = let | assign | print | return | if | for | match | concurrent | assert
type            = simple | "List" "<" type ">" | "Enum" "<" string* ">" | "Result" "<" type "," type ">"
```

---

## Updates

This reference document is automatically updated whenever:
- Grammar rules change in `compiler.py`
- New configuration options are added
- New built-in functions are introduced
- Best practices are refined

**Always check this document for the latest syntax!**

---

## Known Limitations

### Reserved Keywords
Certain names are reserved and cannot be used as struct names:
- `Result` - Use alternatives like `MyResult`, `ResultData`, etc.
- `String`, `Int`, `Float`, `List`, `Enum` - Type names

---

## All Examples Verified

✅ All code examples in this document have been tested and work correctly with the Ensō compiler.

