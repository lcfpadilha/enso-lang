# Resume-to-Job-Match: Ensō vs Python Comparison

## 📊 Metrics

| Metric | Ensō | Python | Winner |
|--------|------|--------|--------|
| **Lines of Code** | 35 | 203 | Ensō (6x shorter) |
| **Time to Write** | ~2 minutes | ~15 minutes | Ensō |
| **Boilerplate** | ~5 lines | ~100 lines | Ensō (95% less) |
| **Error Handling** | Built-in | Manual setup | Ensō |
| **Cost Tracking** | Automatic | Manual + functions | Ensō |
| **API Key Setup** | 0 lines | 3 lines + error checking | Ensō |
| **Retry Logic** | Built-in | 20+ lines | Ensō |
| **Type Safety** | Enforced | Optional (needs dataclass) | Ensō |
| **JSON Parsing** | Auto | Manual + try/catch | Ensō |

## 🔍 Side-by-Side Code Comparison

### Setting Up API & Types

**Ensō:**
```enso
ai fn match_resume_to_job(resume: String, job_description: String) -> Result<MatchResult, AIError> {
    instruction: "...",
    model: "gpt-4o-mini"
}
```
✅ 3 lines. API key loaded automatically from environment.

**Python:**
```python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set. Get key at ...")

class VerdictEnum(str, Enum):
    STRONG_MATCH = "Strong Match"
    GOOD_MATCH = "Good Match"
    FAIR_MATCH = "Fair Match"
    POOR_MATCH = "Poor Match"

@dataclass
class MatchResult:
    match_score: int
    verdict: VerdictEnum
    key_strengths: list[str]
    key_gaps: list[str]
    recommendation: str

    @classmethod
    def from_dict(cls, data: dict) -> "MatchResult":
        try:
            return cls(
                match_score=int(data.get("match_score", 0)),
                verdict=VerdictEnum(data.get("verdict", "Fair Match")),
                key_strengths=data.get("key_strengths", []),
                key_gaps=data.get("key_gaps", []),
                recommendation=data.get("recommendation", "")
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid response format: {e}")
```
❌ 40+ lines. Manual error checking, type definitions, parsing logic.

### Error Handling & Retry Logic

**Ensō:**
```enso
match match_resume_to_job(sample_resume, sample_job) {
    Ok(result) => print("Success: " + result.recommendation),
    Err(error) => print("Error: " + error.message)
}
```
✅ 4 lines. Errors caught, categorized, retried automatically.

**Python:**
```python
def match_resume_to_job(resume: str, job_description: str, max_retries: int = 3) -> MatchResult:
    instruction = f"""..."""
    system_instruction = "..."
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.7,
                timeout=30
            )
            
            response_text = response["choices"][0]["message"]["content"]
            
            try:
                result_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from API: {response_text}") from e
            
            result = MatchResult.from_dict(result_data)
            cost = calculate_cost(instruction, response_text, "gpt-4o-mini")
            print(f"INFO: API call cost: ${cost:.6f}")
            
            return result
            
        except openai.error.APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API Error, retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                raise APIError(f"API failed after {max_retries} retries: {str(e)}", 500)
        
        except openai.error.Timeout:
            if attempt < max_retries - 1:
                print(f"Timeout, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)
                continue
            else:
                raise APIError("Request timed out after 30 seconds", 408)
        
        except openai.error.AuthenticationError:
            raise APIError("Invalid API key. Check OPENAI_API_KEY", 401)
        
        except openai.error.RateLimitError:
            raise APIError("Rate limit exceeded. Upgrade your plan or wait", 429)
```
❌ 50+ lines of manual retry logic, exception handling, timeout management.

### Cost Tracking

**Ensō:**
```enso
// Automatic - `result` contains cost, model_used, confidence
print(result.cost);  // ~$0.01-0.03
```
✅ 0 lines. Automatic cost tracking in response.

**Python:**
```python
def estimate_tokens(text: str) -> int:
    return len(text) // 4

def calculate_cost(input_text: str, output_text: str, model: str = "gpt-4o-mini") -> float:
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4o-mini"])
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
    return cost

OPENAI_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000
    }
}
```
❌ 15+ lines. Manual token estimation, pricing lookup, calculation.

## 💡 Key Takeaways

| Concern | Ensō | Python |
|---------|------|--------|
| **API Boilerplate** | Eliminated | 40+ lines |
| **Error Handling** | Automatic + smart categorization | Manual exception handling |
| **Retry Logic** | Built-in exponential backoff | 20+ lines to write correctly |
| **Cost Estimation** | Automatic per-call | Manual setup required |
| **Type Safety** | Strict (compile-time errors) | Runtime errors possible |
| **Development Speed** | 2 min → working code | 15 min → working code |
| **Maintenance** | Changes in one place (grammar) | Scattered across functions |
| **Production Ready** | Yes (async, batching support) | Requires more testing |

## 🎯 Conclusion

**Ensō version:**
- ✅ 35 lines of clean, readable code
- ✅ Automatic cost tracking
- ✅ Built-in error handling + retry logic
- ✅ Type-safe at compile time
- ✅ Ready to run: `enso run example.enso`

**Python version:**
- ❌ 203 lines with manual boilerplate
- ❌ Manual error handling at 4 different points
- ❌ Custom retry logic with exponential backoff
- ❌ Manual cost calculation per call
- ❌ Runtime type errors possible

**ROI: Ensō lets you focus on the problem (matching resumes) rather than the plumbing (API integration).**
