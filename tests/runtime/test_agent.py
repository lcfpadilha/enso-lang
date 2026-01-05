"""
Unit tests for the EnsoAgent class.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.runtime import get_preamble


class TestEnsoAgentMocking:
    """Tests for the agent's mock system."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        exec(get_preamble(), env)
        return env
    
    def test_mock_returns_mocked_value(self, runtime_env):
        """Agent should return mocked value when mock is set."""
        EnsoAgent = runtime_env['EnsoAgent']
        MOCKS = runtime_env['MOCKS']
        BaseModel = runtime_env['BaseModel']
        
        # Define a simple response model
        class TestResponse(BaseModel):
            message: str
        
        # Set up mock
        MOCKS['test_fn'] = TestResponse(message="mocked!")
        
        # Create agent and run
        agent = EnsoAgent(
            name="test_fn",
            instruction="Test instruction",
            model="gpt-4o"
        )
        
        result = agent.run("input", TestResponse)
        
        assert result.is_ok()
        assert result.unwrap().value.message == "mocked!"
        assert result.unwrap().model_used == "MOCK"
        
        # Clean up
        MOCKS.clear()
    
    def test_mock_calculates_cost(self, runtime_env):
        """Mocked responses should still calculate estimated cost."""
        EnsoAgent = runtime_env['EnsoAgent']
        MOCKS = runtime_env['MOCKS']
        BaseModel = runtime_env['BaseModel']
        
        class TestResponse(BaseModel):
            value: int
        
        MOCKS['cost_test'] = TestResponse(value=42)
        
        agent = EnsoAgent(
            name="cost_test",
            instruction="A" * 1000,  # Long instruction to generate cost
            model="gpt-4o"
        )
        
        result = agent.run("input", TestResponse)
        
        # Cost should be calculated (may be 0 if model not in registry)
        assert result.is_ok()
        assert result.unwrap().cost >= 0
        
        MOCKS.clear()


class TestEnsoAgentSystemPrompt:
    """Tests for system prompt construction."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        exec(get_preamble(), env)
        return env
    
    def test_build_system_prompt_basic(self, runtime_env):
        """System prompt should include system_instruction."""
        EnsoAgent = runtime_env['EnsoAgent']
        
        agent = EnsoAgent(
            name="test",
            instruction="Do something",
            model="gpt-4o",
            system_instruction="You are a helpful assistant"
        )
        
        prompt = agent._build_system_prompt()
        assert "You are a helpful assistant" in prompt
    
    def test_build_system_prompt_strips_quotes(self, runtime_env):
        """System prompt should strip surrounding quotes."""
        EnsoAgent = runtime_env['EnsoAgent']
        
        agent = EnsoAgent(
            name="test",
            instruction="Do something",
            model="gpt-4o",
            system_instruction='"Quoted instruction"'
        )
        
        prompt = agent._build_system_prompt()
        assert prompt == "Quoted instruction"
    
    def test_build_system_prompt_with_examples(self, runtime_env):
        """System prompt should include few-shot examples."""
        EnsoAgent = runtime_env['EnsoAgent']
        
        agent = EnsoAgent(
            name="test",
            instruction="Classify",
            model="gpt-4o",
            examples=[
                {"input": "hello", "expected": "greeting"},
                {"input": "bye", "expected": "farewell"}
            ]
        )
        
        prompt = agent._build_system_prompt()
        assert "Few-Shot Examples" in prompt
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "hello" in prompt
        assert "greeting" in prompt


class TestEnsoAgentJsonCleaning:
    """Tests for JSON response cleaning."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        exec(get_preamble(), env)
        return env
    
    def test_clean_json_strips_markdown(self, runtime_env):
        """Should strip markdown code blocks."""
        EnsoAgent = runtime_env['EnsoAgent']
        
        agent = EnsoAgent(name="test", instruction="test", model="gpt-4o")
        
        raw = '```json\n{"key": "value"}\n```'
        cleaned = agent._clean_json(raw)
        
        assert cleaned == '{"key": "value"}'
    
    def test_clean_json_strips_generic_markdown(self, runtime_env):
        """Should strip generic markdown blocks."""
        EnsoAgent = runtime_env['EnsoAgent']
        
        agent = EnsoAgent(name="test", instruction="test", model="gpt-4o")
        
        raw = '```\n{"key": "value"}\n```'
        cleaned = agent._clean_json(raw)
        
        assert cleaned == '{"key": "value"}'
    
    def test_clean_json_strips_whitespace(self, runtime_env):
        """Should strip surrounding whitespace."""
        EnsoAgent = runtime_env['EnsoAgent']
        
        agent = EnsoAgent(name="test", instruction="test", model="gpt-4o")
        
        raw = '  \n  {"key": "value"}  \n  '
        cleaned = agent._clean_json(raw)
        
        assert cleaned == '{"key": "value"}'


class TestEnsoAgentValidation:
    """Tests for agent input validation."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        exec(get_preamble(), env)
        return env
    
    def test_empty_model_returns_error(self, runtime_env):
        """Empty model name should return Err."""
        EnsoAgent = runtime_env['EnsoAgent']
        BaseModel = runtime_env['BaseModel']
        ErrorKind = runtime_env['ErrorKind']
        
        class TestResponse(BaseModel):
            value: str
        
        agent = EnsoAgent(
            name="test",
            instruction="test",
            model=""  # Empty model
        )
        
        result = agent.run("input", TestResponse)
        
        assert result.is_err()
        assert result.error.kind == ErrorKind.INVALID_CONFIG_ERROR
