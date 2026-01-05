# ==========================================
# ERROR HANDLING: Result<T, E> Model
# ==========================================

class ErrorKind(str, Enum):
    """Categorizes AI failures for composable error handling."""
    API_ERROR = "ApiError"
    PARSE_ERROR = "ParseError"
    HALLUCINATION_ERROR = "HallucinationError"
    TIMEOUT_ERROR = "TimeoutError"
    REFUSAL_ERROR = "RefusalError"
    INVALID_CONFIG_ERROR = "InvalidConfigError"


class AIError(BaseModel):
    """Rich error context for AI operations."""
    kind: ErrorKind
    message: str
    details: Optional[str] = None
    cost: float
    model: str
    timestamp: Optional[str] = None
    
    def __str__(self):
        result = "❌ " + str(self.kind) + ": " + str(self.message)
        if self.details:
            result = result + chr(10) + "   Details: " + str(self.details)
        if self.model:
            result = result + chr(10) + "   Model: " + str(self.model)
        if self.cost and self.cost > 0:
            result = result + chr(10) + "   Cost: $" + str(round(self.cost, 6))
        return result


class Result:
    """Base class for Result<T, E> (Ok or Err)."""
    
    def is_ok(self) -> bool:
        return isinstance(self, Ok)
    
    def is_err(self) -> bool:
        return isinstance(self, Err)
    
    def unwrap(self):
        """Get value or raise error."""
        if isinstance(self, Ok):
            return self.value
        else:
            raise RuntimeError(f"Called unwrap() on Err: {self.error.message}")
    
    def unwrap_or(self, default):
        """Get value or return default."""
        if isinstance(self, Ok):
            return self.value
        else:
            return default


class Ok(Result):
    """Success case: Ok<T>."""
    
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"Ok({self.value})"


class Err(Result):
    """Error case: Err<E>."""
    
    def __init__(self, error):
        self.error = error
    
    def __repr__(self):
        return str(self.error)
    
    def __str__(self):
        return str(self.error)
