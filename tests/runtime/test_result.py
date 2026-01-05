"""
Unit tests for the Result type system (Ok, Err, AIError).
"""
import pytest
import sys
import os

# Add the runtime modules to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# We need to set up the imports that would normally be in preamble_header
from typing import Any, Optional, Union
from pydantic import BaseModel
from enum import Enum

# Now we can exec the result.py content to test it
from core.runtime import get_preamble


class TestResultType:
    """Tests for Ok and Err result types."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        # Execute the preamble to get Ok, Err, Result, etc.
        exec(get_preamble(), env)
        return env
    
    def test_ok_is_ok(self, runtime_env):
        """Ok.is_ok() should return True."""
        Ok = runtime_env['Ok']
        result = Ok("success")
        assert result.is_ok() is True
        assert result.is_err() is False
    
    def test_err_is_err(self, runtime_env):
        """Err.is_err() should return True."""
        Err = runtime_env['Err']
        AIError = runtime_env['AIError']
        ErrorKind = runtime_env['ErrorKind']
        
        error = AIError(
            kind=ErrorKind.API_ERROR,
            message="test error",
            cost=0.0,
            model="test"
        )
        result = Err(error)
        assert result.is_err() is True
        assert result.is_ok() is False
    
    def test_ok_unwrap(self, runtime_env):
        """Ok.unwrap() should return the value."""
        Ok = runtime_env['Ok']
        result = Ok(42)
        assert result.unwrap() == 42
    
    def test_err_unwrap_raises(self, runtime_env):
        """Err.unwrap() should raise RuntimeError."""
        Err = runtime_env['Err']
        AIError = runtime_env['AIError']
        ErrorKind = runtime_env['ErrorKind']
        
        error = AIError(
            kind=ErrorKind.API_ERROR,
            message="test error",
            cost=0.0,
            model="test"
        )
        result = Err(error)
        
        with pytest.raises(RuntimeError, match="Called unwrap"):
            result.unwrap()
    
    def test_ok_unwrap_or(self, runtime_env):
        """Ok.unwrap_or() should return the value, not default."""
        Ok = runtime_env['Ok']
        result = Ok("value")
        assert result.unwrap_or("default") == "value"
    
    def test_err_unwrap_or(self, runtime_env):
        """Err.unwrap_or() should return the default."""
        Err = runtime_env['Err']
        AIError = runtime_env['AIError']
        ErrorKind = runtime_env['ErrorKind']
        
        error = AIError(
            kind=ErrorKind.API_ERROR,
            message="test error",
            cost=0.0,
            model="test"
        )
        result = Err(error)
        assert result.unwrap_or("default") == "default"


class TestAIError:
    """Tests for AIError formatting."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        exec(get_preamble(), env)
        return env
    
    def test_ai_error_str_basic(self, runtime_env):
        """AIError string should include kind and message."""
        AIError = runtime_env['AIError']
        ErrorKind = runtime_env['ErrorKind']
        
        error = AIError(
            kind=ErrorKind.API_ERROR,
            message="Connection failed",
            cost=0.0,
            model="gpt-4o"
        )
        error_str = str(error)
        assert "API_ERROR" in error_str
        assert "Connection failed" in error_str
    
    def test_ai_error_str_with_details(self, runtime_env):
        """AIError string should include details when present."""
        AIError = runtime_env['AIError']
        ErrorKind = runtime_env['ErrorKind']
        
        error = AIError(
            kind=ErrorKind.PARSE_ERROR,
            message="Invalid JSON",
            details="Unexpected token at position 5",
            cost=0.001,
            model="gpt-4o"
        )
        error_str = str(error)
        assert "Details:" in error_str
        assert "Unexpected token" in error_str
    
    def test_ai_error_str_with_cost(self, runtime_env):
        """AIError string should include cost when > 0."""
        AIError = runtime_env['AIError']
        ErrorKind = runtime_env['ErrorKind']
        
        error = AIError(
            kind=ErrorKind.API_ERROR,
            message="Error",
            cost=0.005,
            model="gpt-4o"
        )
        error_str = str(error)
        assert "Cost:" in error_str


class TestErrorKind:
    """Tests for ErrorKind enum values."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        exec(get_preamble(), env)
        return env
    
    def test_error_kinds_exist(self, runtime_env):
        """All expected ErrorKind values should exist."""
        ErrorKind = runtime_env['ErrorKind']
        
        expected_kinds = [
            'API_ERROR',
            'PARSE_ERROR',
            'HALLUCINATION_ERROR',
            'TIMEOUT_ERROR',
            'REFUSAL_ERROR',
            'INVALID_CONFIG_ERROR'
        ]
        
        for kind in expected_kinds:
            assert hasattr(ErrorKind, kind), f"ErrorKind.{kind} should exist"
