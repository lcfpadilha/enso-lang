# ==========================================
# LLM DRIVERS
# ==========================================

class LLMDriver(ABC):
    """Abstract base class for LLM API drivers."""
    
    @abstractmethod
    def complete(self, model: str, system: str, user: str, schema: Dict) -> str:
        pass


class OpenAIDriver(LLMDriver):
    """Driver for OpenAI API (GPT models)."""
    
    def complete(self, model: str, system: str, user: str, schema: Dict) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY. "
                "Set it with: export OPENAI_API_KEY='your-key-here'. "
                "Get key at: https://platform.openai.com/api-keys"
            )
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        system_prompt = f"{system}\n\nYou MUST return valid JSON matching this schema:\n{json.dumps(schema)}"
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            raise TimeoutError("OpenAI API request timed out (30s). Try again or check your network.")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to OpenAI API. Check your internet connection and API endpoint.")
        except requests.exceptions.HTTPError as e:
            status = resp.status_code if 'resp' in locals() else "unknown"
            if status == 401:
                raise ValueError("Invalid OpenAI API key. Check OPENAI_API_KEY environment variable.")
            elif status == 429:
                raise Exception(
                    "OpenAI rate limit exceeded. "
                    "Wait a minute before retrying or upgrade at https://platform.openai.com/account/billing/overview"
                )
            elif status == 500:
                raise Exception("OpenAI server error (500). Their service may be down. Try again in a moment.")
            else:
                error_detail = resp.text if 'resp' in locals() else str(e)
                raise Exception(f"OpenAI API error ({status}): {error_detail}")
        except Exception as e:
            if 'resp' in locals() and hasattr(resp, 'text'):
                print(f"    [Details] {resp.text}", file=sys.stderr)
            raise e


class GeminiDriver(LLMDriver):
    """Driver for Google Gemini API."""
    
    def complete(self, model: str, system: str, user: str, schema: Dict) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY. "
                "Set it with: export GEMINI_API_KEY='your-key-here'. "
                "Get key at: https://aistudio.google.com/app/apikey"
            )
        
        api_key = api_key.strip()
        clean_model = model.replace('"', '').replace("'", "").strip()
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": f"System: {system}\nUser: {user}"}]}],
            "generationConfig": {
                "response_mime_type": "application/json", 
                "response_schema": schema
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
        except requests.exceptions.Timeout:
            raise TimeoutError("Gemini API request timed out (30s). Try again or check your network.")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to Gemini API. Check your internet connection and API endpoint.")
        except requests.exceptions.HTTPError as e:
            status = resp.status_code if 'resp' in locals() else "unknown"
            if status == 400:
                raise ValueError(
                    "Gemini API error: Invalid request. Check model name and schema. "
                    "Valid models: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash"
                )
            elif status == 401:
                raise ValueError("Invalid Gemini API key. Check GEMINI_API_KEY environment variable.")
            elif status == 403:
                raise ValueError(
                    "Gemini API access denied. Check: API enabled in GCP Console, valid API key, region restrictions."
                )
            elif status == 429:
                raise Exception(
                    "Gemini free tier rate limit exceeded (60 requests/min). "
                    "Consider waiting or upgrading at https://ai.google.dev/pricing"
                )
            elif status == 500:
                raise Exception("Gemini API server error (500). Their service may be down. Try again in a moment.")
            else:
                error_detail = resp.text if 'resp' in locals() else str(e)
                raise Exception(f"Gemini API error ({status}): {error_detail}")
        except Exception as e:
            if 'resp' in locals() and hasattr(resp, 'text'):
                try:
                    print(f"    [Details] {resp.text}", file=sys.stderr)
                except:
                    pass
            raise e


class LocalDriver(LLMDriver):
    """Driver for local Ollama models."""
    
    def complete(self, model: str, system: str, user: str, schema: Dict) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3", 
            "prompt": f"{system}\nSchema: {json.dumps(schema)}\nUser: {user}", 
            "format": "json", 
            "stream": False
        }
        resp = requests.post(url, json=payload)
        return resp.json()['response']


def get_driver(driver_type):
    """Factory function to get the appropriate LLM driver."""
    if driver_type == "gemini":
        return GeminiDriver()
    if driver_type == "local":
        return LocalDriver()
    return OpenAIDriver()
