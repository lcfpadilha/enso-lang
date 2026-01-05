# Python Comparison Examples

**Ensō vs Python: See the difference.**

This folder contains production-ready Ensō examples paired with equivalent Python implementations, demonstrating the dramatic reduction in code complexity.

---

## 🎯 Resume-to-Job Match

Match a resume against a job description and get a compatibility score with reasoning.

### Files

| File | Lines | Description |
|------|-------|-------------|
| [resume_job_match.enso](resume_job_match.enso) | 35 | Clean, declarative Ensō |
| [resume_job_match.py](resume_job_match.py) | 203 | Equivalent Python with all boilerplate |

### Run It

```bash
# Ensō version
enso run python-comparison-examples/resume_job_match.enso

# Python version (requires OPENAI_API_KEY)
python python-comparison-examples/resume_job_match.py
```

**Output:**
```
Match Score: 92
Verdict: Strong Match
Recommendation: Hire immediately - exceptional fit for the role
```

---

## 📊 Metrics

| Metric | Ensō | Python |
|--------|------|--------|
| **Lines of Code** | 35 | 203 |
| **Time to Write** | ~2 min | ~15 min |
| **Boilerplate** | ~5 lines | ~100 lines |
| **Error Handling** | Built-in | Manual (50+ lines) |
| **Cost Tracking** | Automatic | Manual setup |
| **Retry Logic** | Built-in | 20+ lines |
| **Type Safety** | Compile-time | Runtime only |

---

## 🔍 Code Comparison

### API & Types Setup

**Ensō** (3 lines):
```enso
ai fn match_resume_to_job(resume: String, job_description: String) -> Result<MatchResult, AIError> {
    instruction: "...",
    model: "gemini-2.0-flash"
}
```

**Python** (40+ lines):
```python
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set")

class VerdictEnum(str, Enum):
    STRONG_MATCH = "Strong Match"
    # ... more enum values

@dataclass
class MatchResult:
    match_score: int
    verdict: VerdictEnum
    key_strengths: list[str]
    key_gaps: list[str]
    recommendation: str

    @classmethod
    def from_dict(cls, data: dict) -> "MatchResult":
        # ... parsing logic with error handling
```

### Error Handling

**Ensō** (4 lines):
```enso
match match_resume_to_job(sample_resume, sample_job) {
    Ok(result) => print(result.recommendation),
    Err(error) => print(error.message)
}
```

**Python** (50+ lines):
```python
for attempt in range(max_retries):
    try:
        response = openai.ChatCompletion.create(...)
        result_data = json.loads(response_text)
        return MatchResult.from_dict(result_data)
    except openai.error.APIError as e:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
            continue
        raise APIError(f"Failed after {max_retries} retries")
    except openai.error.Timeout:
        # ... more exception handling
    except openai.error.AuthenticationError:
        # ... more exception handling
```

### Cost Tracking

**Ensō** (0 lines - automatic):
```enso
print(result.cost);  // Built into every response
```

**Python** (15+ lines):
```python
def estimate_tokens(text: str) -> int:
    return len(text) // 4

def calculate_cost(input_text: str, output_text: str, model: str) -> float:
    pricing = OPENAI_PRICING.get(model)
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
```

---

## 💡 Adapt for Your Use Case

```bash
# 1. Copy the template
cp python-comparison-examples/resume_job_match.enso my-example.enso

# 2. Edit with your logic
# 3. Run it
enso run my-example.enso
```

---

## Questions?

See the main [README.md](../README.md) for syntax and CLI commands.
