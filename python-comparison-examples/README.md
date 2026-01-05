# Python Comparison Examples

This folder contains **production-ready Ensō examples paired with equivalent Python implementations**. Each example demonstrates the dramatic difference in code complexity, development time, and maintainability between Ensō and traditional Python.

## 📊 Structure

Each example has its own folder containing:
- **`main.enso`** – Clean, declarative Ensō implementation
- **`main.py`** – Equivalent Python implementation with all boilerplate
- **`COMPARISON.md`** – Side-by-side code analysis and metrics

## 🎯 Resume-to-Job Match

The flagship example demonstrating Ensō's value proposition.

### What It Does
Match a resume against a job description and get a compatibility score with reasoning.

### Run the Ensō Version

```bash
enso run python-comparison-examples/resume-job-match/main.enso
```

**Output:**
```
Match Score: 92
Verdict: Strong Match
Recommendation: Hire immediately - exceptional fit for the role
```

### Key Metrics

| Metric | Ensō | Python |
|--------|------|--------|
| **Lines of Code** | 35 | 203 |
| **Development Time** | ~2 min | ~15 min |
| **Error Handling** | Built-in | Manual (50+ lines) |
| **Cost Tracking** | Automatic | Manual + functions |
| **Retry Logic** | Built-in exponential backoff | 20+ lines to implement |
| **Type Safety** | Compile-time checks | Runtime only |

### What You'll Learn

1. **Ensō version** – How to declare AI logic cleanly
2. **Python version** – All the boilerplate you avoid with Ensō
3. **Comparison** – Metrics, code analysis, and ROI breakdown

### See the Comparison

```bash
cat python-comparison-examples/resume-job-match/COMPARISON.md
```

---

## 🔄 How Comparisons Work

Each comparison shows:

1. **API & Type Setup** – How much code just to initialize the client
2. **Error Handling & Retry Logic** – The complexity of production-grade error handling
3. **Cost Tracking** – Manual cost calculation vs automatic
4. **Main Logic** – The core business logic (similar in both, but context is different)

**The Takeaway:** Ensō lets you focus on the problem. Python forces you to focus on the plumbing.

---

## 🚀 Future Comparisons

Planned additions:
- [ ] **Invoice Extraction** – Document processing with cost comparison
- [ ] **Content Moderation** – Batch processing pipeline
- [ ] **LLM Routing** – Multi-model selection logic
- [ ] **Concurrent Batch Jobs** – Parallel processing with error collection

---

## 💡 How to Adapt These Examples

1. Copy the folder: `cp -r python-comparison-examples/resume-job-match my-example`
2. Modify `main.enso` with your logic
3. Run it: `enso run my-example/main.enso`
4. Use `python-comparison-examples/resume-job-match/main.py` as reference if you need to understand the Python equivalent

---

## Questions?

See the main [README.md](../README.md) for syntax, CLI commands, and feature overview.
