"""
Resume-to-Job-Match (Python Version)
====================================
Equivalent to the Ensō version but written in pure Python.
Notice: 3x more code, more boilerplate, more error handling to write manually.
"""

import os
import json
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import openai
import time

# ========== Setup API Client ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set. Get key at https://platform.openai.com/api-keys")

# ========== Type Definitions ==========
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
        """Parse API response into typed MatchResult"""
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

# ========== Error Handling ==========
class APIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(f"API Error ({status_code}): {message}")

# ========== Cost Tracking ==========
OPENAI_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,   # $0.15 per 1M input tokens
        "output": 0.60 / 1_000_000   # $0.60 per 1M output tokens
    }
}

def estimate_tokens(text: str) -> int:
    """Rough estimate: 4 chars = 1 token"""
    return len(text) // 4

def calculate_cost(input_text: str, output_text: str, model: str = "gpt-4o-mini") -> float:
    """Calculate API call cost"""
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4o-mini"])
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    
    cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
    return cost

# ========== API Call with Retries & Error Handling ==========
def match_resume_to_job(resume: str, job_description: str, max_retries: int = 3) -> MatchResult:
    """
    Match a resume against a job description.
    Handles: API errors, timeouts, parsing, cost tracking.
    """
    
    instruction = f"""You are a hiring manager evaluating candidate fit.

Resume:
{resume}

Job Description:
{job_description}

Analyze how well this candidate matches the role. Return a match score (0-100), verdict, top 3 strengths, top 3 gaps, and a recommendation.

IMPORTANT: Respond with valid JSON only, no extra text. Format:
{{
    "match_score": <int>,
    "verdict": "<Strong Match|Good Match|Fair Match|Poor Match>",
    "key_strengths": ["skill1", "skill2", "skill3"],
    "key_gaps": ["gap1", "gap2", "gap3"],
    "recommendation": "<text>"
}}"""
    
    system_instruction = "You are a hiring manager evaluating candidate fit."
    
    # Retry loop for transient failures
    for attempt in range(max_retries):
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.7,
                timeout=30
            )
            
            # Extract response text
            response_text = response["choices"][0]["message"]["content"]
            
            # Parse JSON response
            try:
                result_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from API: {response_text}") from e
            
            # Convert to typed result
            result = MatchResult.from_dict(result_data)
            
            # Calculate and track cost
            cost = calculate_cost(instruction, response_text, "gpt-4o-mini")
            print(f"INFO: API call cost: ${cost:.6f}")
            
            return result
            
        except openai.error.APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
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

# ========== Main ==========
def main():
    sample_resume = "John Doe. 8 years of software engineering experience. Expert in Python, Go, Kubernetes. Led team of 5 engineers. Built distributed payment systems handling 1M+ transactions/day."
    
    sample_job = "Senior Backend Engineer - fintech startup. 5-10 years exp required. Must know: Python, Go, distributed systems, SQL. Will lead backend team. Salary: $200-250k."
    
    try:
        result = match_resume_to_job(sample_resume, sample_job)
        print(f"Match Score: {result.match_score}")
        print(f"Verdict: {result.verdict}")
        print(f"Recommendation: {result.recommendation}")
    except APIError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Validation Error: {e}")

if __name__ == "__main__":
    main()
