# The Ensō Manifesto
### Philosophy for the Age of Native Intelligence

Software engineering has shifted. Intelligence is no longer an algorithm we write; it is a service we call. Yet, our programming languages treat Artificial Intelligence as an external chaotic dependency—a library to be imported, managed, and feared.

**We believe that Intelligence should be a primitive, not a plugin.**

We value:

* **Declarative Intent** over **Imperative Glue Code**.
* **Typed Contracts** over **Prompt Engineering**.
* **Deterministic Mocking** over **Live API Dependency**.
* **Compiler Guarantees** over **Runtime Parsers**.

### The Core Principles

**1. The Prompt is the Code**
We do not hide prompts in strings inside Python dictionaries. The instruction *is* the function body. The compiler optimizes it, sends it, and handles the retry logic.

**2. Types are Instructions**
A Type definition (`struct`) is not just for validation; it is context for the model. When you define a `struct`, you are teaching the AI how to think.

**3. Determinism in Testing**
An AI application that cannot be tested offline is broken. We treat `mock` as a first-class citizen, allowing developers to test logic flows without spending a cent or waiting for an API call.

**4. Model Agnosticism**
Your business logic should not depend on `import openai` or `import google.generativeai`. The specific model (GPT-4, Gemini, Claude) is a configuration detail, not an implementation detail.