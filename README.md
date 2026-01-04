# Ensō (円相)

**The AI-Native Systems Language.**

Ensō is a statically-typed, compiled language designed for the Agentic Era. It treats Large Language Models (LLMs) as reliable CPU functions, abstracting away the complexity of API clients, JSON schema generation, retry logic, and structured parsing.

**Write Intent. Compile to Infrastructure.**

---

## ⚡ Quick Start

### 1. The Code (`hiring.enso`)

```rust
// 1. Define your data shape (The "Contract")
struct Skill {
    name: String,
    level: Enum<"Beginner", "Intermediate", "Expert">
}

struct Candidate {
    name: String,
    skills: List<Skill>
}

// 2. Define your Intelligence (The "Declarative Logic")
ai fn extract_resume(text: String) -> Result<Candidate, AIError> {
    instruction: "Extract name and skills. Be strict with seniority levels."
    model: "gemini-flash-latest"
}

// 3. Define your Orchestration (The "CPU Logic")
fn main() {
    let raw_text = "Hi, I'm Neo. I know Kung Fu (Expert) and Python (Beginner).";
    
    // The 'match' statement guarantees you handle success AND failure.
    match extract_resume(raw_text) {
        Ok(candidate) => {
            print("Hired: " + candidate.value.name);
        },
        Err(error) => {
            print("Parsing failed. Reason:");
            print(error.message);
        }
    }
}
```

### 2. Run It

```bash
enso run hiring.enso
```

### Output

```text
--- Parsing ---
[Enso] Agent 'extract_resume' -> gemini-flash-latest...
    [Meta] Cost: $0.00012 | Latency: 1.2s
Name: Neo
Skill: Kung Fu
Skill: Python
```

### Learn the Syntax

👉 **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Complete grammar guide with all syntax rules, types, functions, and best practices. **Start here for syntax questions.**

---

## 📚 Cookbook Examples

The `examples/` folder contains **production-ready examples** that serve as a cookbook for common use cases. Each example demonstrates real-world patterns you can adapt for your own projects:

| Example | Focus | Use Cases |
|---------|-------|----------|
| **1. Structured Data Extraction** | Type-safe extraction, batch processing | Resume parsing, form filling, data migration, document processing |
| **2. Classification & Decision Making** | Enums, constrained outputs, moderation | Sentiment analysis, content moderation, email routing, ticket classification |
| **3. Large Context Windows** | Long document processing, web agents | Document analysis, competitive intelligence, log analysis, research |
| **4. Complete Hiring Workflow** | Full pipeline, multiple AI functions | End-to-end hiring automation, multi-step decision making |

**Try them all:**
```bash
enso run examples/1_structured_data_extraction.enso
enso run examples/2_classification_decision_making.enso
enso run examples/3_large_context_web_agent.enso
enso run examples/4_complete_hiring_workflow.enso
```

Each example includes:
- ✅ Real-world scenario
- ✅ Production-ready code
- ✅ Comments explaining the pattern
- ✅ Runnable with `enso.py run` command

---

## Why Ensō?

### 1. Type-Safe Intelligence
Ensō compiles strict types into **JSON Schemas** automatically. If the LLM returns data that doesn't match your `struct`, Ensō catches it before it hits your logic.

### 2. Error as a First-Class Citizen
AI is non-deterministic. It *will* fail. Ensō's `Result<T, E>` type and `match` syntax force you to handle refusals, timeouts, and hallucinations at compile time. No more crashing in production because the model returned "I cannot answer that."

### 3. Vendor Agnostic
Switch from `gemini-1.5-flash` to `gpt-4o` by changing one line of code. The compiler handles the API differences.

---

## 🛠 Features

### 🧠 AI Functions (ai fn)

Forget import openai. Define a function, give it an instruction, and declare what you want back. The compiler handles the rest.

```rust
ai fn summarize(text: String) -> String {
    instruction: "Summarize in 3 bullet points."
}
```

### 🔒 Structured Types (`struct`)

Ensō compiles strict types into **Pydantic Models** and automatically generates **JSON Schemas** to force the LLM to return valid data.

```rust
struct User {
    name: String,
    age: Int,
    tags: List<String>
}
```

### 🧪 Native Testing & Mocking

Test your logic without hitting the API. Ensō has a built-in test runner with `mock` support.

```rust
test "User parsing logic" {
    // Force the AI to return this specific data
    mock extract_resume => Candidate { name: "Trinity", skills: [] };
    
    let res = extract_resume("irrelevant text");
    assert res.value.name == "Trinity";
}
```

### 🔌 Standard CPU Functions

Write standard imperative code to glue your AI steps together.

```rust
fn process(data: String) -> Int {
    if data == "" {
        return 0;
    }
    let parsed = ai_func(data);
    return parsed.value.score;
}
```

### ⚙️ Concurrent Batch Operations

Process multiple items through an AI function in parallel. Returns `List<Result<T, E>>` with both successes and failures.

```rust
ai fn analyze(text: String) -> Result<Sentiment, AIError> {
    instruction: "Analyze sentiment for text",
    model: "gpt-4o"
}

test "Batch Processing" {
    let texts = ["Great day", "Bad timing", "It is okay"];
    concurrent analyze(text) for text in texts;  // Process all in parallel
    
    // Results: List<Result<Sentiment, AIError>>
    for result in analyze_results {
        match result {
            Ok(sentiment_data) => {
                print(sentiment_data.value.mood);
            },
            Err(error) => {
                print(error.message);
            }
        }
    }
}
```

**Key Features:**
- ✅ True parallel execution using `asyncio.gather()`
- ✅ Composable error handling (collects both successes and failures)
- ✅ Backward compatible with single-function calls
- ✅ Full mock support for testing

---

## 🚀 Prompt Engineering Superpowers

Ensō  includes powerful features for advanced prompt engineering, enabling you to build AI systems with more accuracy.

### 1. Variable Interpolation

Inject function parameters directly into your prompts:

```rust
ai fn extract(resume: String, target_role: String) -> Candidate {
    instruction: "Extract {target_role} candidate from resume: {resume}",
    model: "gpt-4o"
}

// Automatically becomes:
// "Extract Senior Backend Engineer candidate from resume: John: 10y Python..."
```

### 2. System Instructions

Separate role-based context from task instructions:

```rust
ai fn assess_fit(resume: String, company_culture: String) -> Candidate {
    system_instruction: "You are a world-class technical recruiter with 20 years of experience.",
    instruction: "Assess fit for our {company_culture} team. Resume: {resume}",
    model: "gpt-4o"
}
```

### 3. Few-Shot Examples

Include examples to dramatically improve accuracy:

```rust
ai fn match_candidate(resume: String, seniority: String) -> Candidate {
    system_instruction: "You are a top technical recruiter.",
    instruction: "Match {seniority} candidate. Resume: {resume}",
    
    // Few-shot examples guide the LLM
    examples: [
        (
            resume: "John: 10y Python, led 3 teams, mentored 5+ engineers",
            seniority: "Senior",
            expected: Candidate { 
                name: "John", 
                match_score: 92 
            }
        ),
        (
            resume: "Jane: Fresh grad, bootcamp, 1 toy project",
            seniority: "Senior",
            expected: Candidate { 
                name: "Jane", 
                match_score: 15 
            }
        )
    ],
    
    temperature: 0.1,  // Low temperature for structured outputs
    model: "gpt-4o"
}
```

---

## 🏗 Architecture

Ensō is a **Transpiled Language**.

1. **Source Code (`.enso`):** Your clean, readable logic.
2. **Compiler:** Uses Lark to parse the AST.
3. **Transpilation:** Converts Ensō AST into Python + Pydantic + Requests.
4. **Runtime:** The generated Python code is executed, managing API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`) and HTTP connections automatically.

---

## 📦 Installation & Setup

### 1. Clone & Install Dependencies

```bash
git clone [https://github.com/your-repo/enso.git](https://github.com/your-repo/enso.git)
cd enso
```

### 2. Configure API Keys

```bash
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

### 3. Setup the enso Command

Run the following command in the project root to install the CLI tool in editable mode. This makes enso globally available while allowing you to modify the source code.

```bash
pip install -e .
```

### Initialize a Project

```bash
enso init
```

## 💻 CLI Usage

| Command | Description | Example |
| :--- | :--- | :--- |
| `enso init` | Creates a new project with `main.enso` | `enso init` |
| `enso run` | Compiles and executes a file | `enso run main.enso` |
| `enso test` | Runs internal tests (mocks only) | `enso test main.enso` |
| `enso test --include_ai` | Runs tests allowing real AI calls | `enso test main.enso --include_ai` |
| `enso update` | Updates local model pricing/registry | `enso update` |

---

## 🎨 Syntax Highlighting

### VS Code

Enso files (`.enso`) have full syntax highlighting support in VS Code.

**Installation:**
1. Open VS Code **Extensions** panel (`Ctrl+Shift+X`)
2. Click the menu button (⋯) → **"Install from VSIX..."**
3. Navigate to `enso-vscode/enso-language-support-0.1.0.vsix` in the project root
4. Click **"Install"**
5. Reload VS Code (`Ctrl+Shift+P` → "Reload Window")

---

## 📋 Phase 2: Future Roadmap

These features are designed but deferred for post-MVP iterations:

### Result Type Enhancements
- [ ] **Monadic Operations** – Add `bind()` / `flatMap()` for composing Result-returning operations
  - Enables: `analyze(text).bind(classify).bind(score)` for chaining without explicit matching
- [ ] **Advanced Type Checks** – Better pattern matching with guards
  - Enables: `result.is_ok()`

### Concurrency Improvements
- [ ] **Per-Item Timeouts** – Configure timeout per concurrent item
- [ ] **Progress Callbacks** – Monitor long-running batch operations
- [ ] **Automatic Retry Logic** – Retry failed items with exponential backoff
- [ ] **Rate Limiting** – Control QPS to avoid API throttling

### Library Generation
- [ ] **`pub fn` Visibility** – Mark functions for export vs. internal-only
- [ ] **Module Hierarchy** – Preserve directory structure in generated package
  - `src/sentiments.enso` → `ai_lib.sentiments.analyze()`
- [ ] **Auto-Generated Docstrings** – Extract from instructions for IDE tooltips
- [ ] **Type Stubs (.pyi)** – For better IDE autocomplete and type checking

### Language Features
- [x] **✅ Variable Interpolation in Prompts** – `"Analyze {text} with context {context}"` (DONE)
- [x] **✅ System Instructions** – Separate role context from task instructions (DONE)
- [x] **✅ Few-Shot Examples** – Guide LLM with examples for better accuracy (DONE)
- [ ] **Multi-Model Routing** – Route based on cost/latency/capability
  - `model: select("gpt-4o" if complex else "gpt-4o-mini")`
- [ ] **Streaming Responses** – Stream tokens for real-time feedback
- [ ] **Prompt Caching** – Reuse expensive prompt prefixes across calls
- [ ] **Tool Use / Function Calling** – Define functions for LLM to call back

### Developer Experience
- [ ] **SDK Generation** – Generate TypeScript/Go SDKs from Ensō definitions
- [ ] **Observability** – Built-in tracing, cost reporting, analytics
- [ ] **Local Development** – Mock LLM responses with recorded interactions
- [ ] **Hot Reload** – Modify Ensō files and restart without recompiling

### Performance & Reliability
- [ ] **Compiled Binaries** – Compile to native binaries (PyO3 / maturin)
- [ ] **Better Error Messages** – With suggestions and error codes
- [ ] **Cost Forecasting** – Estimate total cost before running batch
- [ ] **Circuit Breaker** – Fail-safe when errors exceed threshold

---

*Ensō is currently in Alpha. Built for the stress test of the future.*
