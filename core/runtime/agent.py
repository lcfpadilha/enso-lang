# ==========================================
# ENSO AGENT
# ==========================================

class Probabilistic(BaseModel):
    """Wrapper for AI function results with metadata."""
    value: Any
    confidence: float
    cost: float
    model_used: str


class EnsoAgent:
    """
    Core agent that executes AI functions.
    
    Handles:
    - Model configuration and driver selection
    - System prompt construction with few-shot examples
    - Mock injection for testing
    - Cost tracking and analysis
    - Error categorization
    """
    
    def __init__(self, name, instruction, model="gpt-4o", system_instruction=None, examples=None, temperature=None, **kwargs):
        self.name = name
        self.instruction = instruction
        self.system_instruction = system_instruction
        self.examples = examples or []
        self.temperature = temperature if temperature is not None else 0.1
        self.model = model
        self.spec = load_model_config(model)
        self.driver = get_driver(self.spec.get('type', 'openai'))
        self.config = kwargs

    def _build_system_prompt(self):
        """Combine system instruction and few-shot examples into complete prompt."""
        parts = []

        if self.system_instruction:
            sys_instr = self.system_instruction
            if sys_instr.startswith('"') or sys_instr.startswith("'"):
                sys_instr = sys_instr[1:]
            if sys_instr.endswith('"') or sys_instr.endswith("'"):
                sys_instr = sys_instr[:-1]
            parts.append(sys_instr)

        if self.examples:
            parts.append("\n\n--- Few-Shot Examples ---")
            for i, example in enumerate(self.examples, 1):
                parts.append(f"\nExample {i}:")
                # Format input fields
                for key, value in example.items():
                    if key != 'expected':
                        parts.append(f"  {key}: {value}")
                # Format expected output
                if 'expected' in example:
                    parts.append(f"  Expected: {example['expected']}")

        return "\n".join(parts)

    def _clean_json(self, raw: str) -> str:
        """Clean LLM response to extract valid JSON."""
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        if raw.startswith("System:"):
            raw = raw.replace("System:", "", 1)
        return raw.strip()

    def run(self, input_text, response_model) -> Union[Ok, Err]:
        """
        Execute AI function and return Result<Probabilistic<T>, AIError>.
        Categorizes all failures as AIError with proper ErrorKind.
        """
        cost = 0.0
        
        # Track in analysis mode (if _ANALYSIS_RESULTS is set, we're in analysis)
        if _ANALYSIS_RESULTS is not None:
            _ANALYSIS_RESULTS["execution_path"].append(f"[CALL] {self.name}({self.model})")
        
        try:
            # Check mocks first
            if self.name in MOCKS:
                print(f"  \033[93m[Mock]\033[0m Serving response for '{self.name}'", file=sys.stderr)
                mocked_value = MOCKS[self.name]
                
                # Calculate cost even for mocks (for analysis purposes)
                # Estimate tokens: approximately 1 token per 3.5 characters
                mock_str = str(mocked_value) if mocked_value else ""
                
                # Token estimation: in = system_instruction + instruction + input_text
                in_tok = int((len(self.system_instruction or "") + len(self.instruction)) / 3.5 + 256)
                out_tok = int(len(mock_str) / 3.5 + 256)
                cost = (in_tok / 1e6 * self.spec.get('cost_in', 0)) + (out_tok / 1e6 * self.spec.get('cost_out', 0))
                
                # Track mock response in analysis with calculated cost
                if _ANALYSIS_RESULTS is not None:
                    _ANALYSIS_RESULTS["ai_calls"].append({
                        "function": self.name,
                        "model": self.model,
                        "cost": cost,
                        "source": "mock"
                    })
                
                return Ok(Probabilistic(value=mocked_value, confidence=1.0, cost=cost, model_used="MOCK"))

            # Validate configuration
            if not self.model:
                return Err(AIError(
                    kind=ErrorKind.INVALID_CONFIG_ERROR,
                    message="Model name is empty",
                    cost=0.0,
                    model=self.model or "unknown"
                ))

            print("\n\033[92m\033[1mINFO:\033[0m Agent '" + self.name + "' -> " + self.model, file=sys.stderr)
            
            # Try to generate JSON schema
            try:
                raw_schema = response_model.model_json_schema()
                clean_schema = resolve_refs(raw_schema)
            except Exception as e:
                error_msg = str(e)
                if not error_msg or error_msg == "model_json_schema":
                    error_msg = f"{type(e).__name__}: Unable to generate schema for response model"
                return Err(AIError(
                    kind=ErrorKind.INVALID_CONFIG_ERROR,
                    message="Failed to generate response schema",
                    details=error_msg,
                    cost=0.0,
                    model=self.model
                ))
            
            # Build complete system prompt with examples
            system_prompt = self._build_system_prompt()
            final_instruction = f"{system_prompt}\n\n{self.instruction}" if system_prompt else self.instruction

            # Call driver with error handling
            try:
                start_t = time.time()
                raw_json = self.driver.complete(self.model, final_instruction, input_text, clean_schema)
                latency = time.time() - start_t
            except ValueError as e:
                # Invalid config (missing API keys, invalid models, etc)
                error_msg = str(e)
                return Err(AIError(
                    kind=ErrorKind.INVALID_CONFIG_ERROR,
                    message=error_msg,
                    cost=0.0,
                    model=self.model
                ))
            except TimeoutError as e:
                return Err(AIError(
                    kind=ErrorKind.TIMEOUT_ERROR,
                    message="API request timeout",
                    details=str(e),
                    cost=cost,
                    model=self.model
                ))
            except Exception as e:
                # Network/API errors - provide helpful context
                error_msg = str(e)
                
                # Categorize the error for better messaging
                if "rate" in error_msg.lower() or "429" in error_msg:
                    kind = ErrorKind.API_ERROR
                    message = "Rate limit exceeded - too many requests"
                elif "timeout" in error_msg.lower():
                    kind = ErrorKind.TIMEOUT_ERROR
                    message = "Request took too long - try again"
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    kind = ErrorKind.API_ERROR
                    message = "Network connection failed"
                elif "401" in error_msg or "unauthorized" in error_msg.lower():
                    kind = ErrorKind.INVALID_CONFIG_ERROR
                    message = "Invalid API key"
                elif "403" in error_msg or "forbidden" in error_msg.lower():
                    kind = ErrorKind.INVALID_CONFIG_ERROR
                    message = "API access denied - check permissions"
                elif "404" in error_msg or "not found" in error_msg.lower():
                    kind = ErrorKind.INVALID_CONFIG_ERROR
                    message = "Model not found - check model name"
                elif "500" in error_msg:
                    kind = ErrorKind.API_ERROR
                    message = "API server error - try again later"
                else:
                    kind = ErrorKind.API_ERROR
                    message = "API request failed"
                
                return Err(AIError(
                    kind=kind,
                    message=message,
                    details=error_msg[:200],  # Truncate very long error messages
                    cost=cost,
                    model=self.model
                ))
            
            # Parse response
            clean_json = self._clean_json(raw_json)
            # Token estimation: approximately 1 token per 3.5 characters
            system_prompt_len = len(self.system_instruction or "")
            in_tok = int((system_prompt_len + len(self.instruction) + len(input_text)) / 3.5)
            out_tok = int(len(clean_json) / 3.5)
            cost = (in_tok/1e6 * self.spec.get('cost_in', 0)) + (out_tok/1e6 * self.spec.get('cost_out', 0))

            # Try to parse JSON
            try:
                data = json.loads(clean_json)
            except json.JSONDecodeError as e:
                return Err(AIError(
                    kind=ErrorKind.PARSE_ERROR,
                    message="Invalid JSON response",
                    details=f"{str(e)}\nRaw: {raw_json[:200]}",
                    cost=cost,
                    model=self.model
                ))
            
            # Try to validate against schema
            try:
                val = response_model(**data)
                print(f"  \033[96m💰 Cost: ${round(cost, 6)} | ⏱️  Latency: {round(latency, 2)}s\033[0m", file=sys.stderr)
                
                # Track in analysis mode
                if _ANALYSIS_RESULTS is not None:
                    _ANALYSIS_RESULTS["ai_calls"].append({
                        "function": self.name,
                        "model": self.model,
                        "cost": cost,
                        "source": "api"
                    })
                
                return Ok(Probabilistic(value=val, confidence=0.99, cost=cost, model_used=self.model))
            except ValidationError as e:
                return Err(AIError(
                    kind=ErrorKind.HALLUCINATION_ERROR,
                    message="Response doesn't match schema",
                    details=str(e),
                    cost=cost,
                    model=self.model
                ))
        
        except Exception as e:
            # Catch-all for unexpected errors
            return Err(AIError(
                kind=ErrorKind.API_ERROR,
                message="Unexpected error",
                details=str(e),
                cost=cost,
                model=self.model
            ))

    async def run_async(self, input_text, response_model) -> Union[Ok, Err]:
        """Async wrapper for concurrent execution."""
        return self.run(input_text, response_model)

    def run_concurrent(self, items, func_name, response_model):
        """
        Execute agent.run() for each item concurrently using asyncio.
        Returns List[Result[T, E]] (list of Ok or Err for each item).
        """
        async def gather_results():
            tasks = [self.run_async(item, response_model) for item in items]
            return await asyncio.gather(*tasks)
        
        # Run the async function directly with asyncio.run()
        return asyncio.run(gather_results())
