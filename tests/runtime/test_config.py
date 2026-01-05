"""
Unit tests for the configuration utilities.
"""
import pytest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.runtime import get_preamble


class TestResolveRefs:
    """Tests for JSON schema reference resolution."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        exec(get_preamble(), env)
        return env
    
    def test_no_refs(self, runtime_env):
        """Schema without refs should pass through unchanged."""
        resolve_refs = runtime_env['resolve_refs']
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        result = resolve_refs(schema)
        assert result == schema
    
    def test_simple_ref(self, runtime_env):
        """Simple $ref should be replaced with definition."""
        resolve_refs = runtime_env['resolve_refs']
        
        schema = {
            "type": "object",
            "properties": {
                "person": {"$ref": "#/$defs/Person"}
            },
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        result = resolve_refs(schema)
        
        # $ref should be resolved
        assert "$ref" not in str(result)
        # Definition should be inlined
        assert result["properties"]["person"]["type"] == "object"
        assert result["properties"]["person"]["properties"]["name"]["type"] == "string"
        # $defs should be removed
        assert "$defs" not in result
    
    def test_nested_refs(self, runtime_env):
        """Nested $refs should be recursively resolved."""
        resolve_refs = runtime_env['resolve_refs']
        
        schema = {
            "type": "object",
            "properties": {
                "team": {"$ref": "#/$defs/Team"}
            },
            "$defs": {
                "Team": {
                    "type": "object",
                    "properties": {
                        "leader": {"$ref": "#/$defs/Person"}
                    }
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        result = resolve_refs(schema)
        
        # Both refs should be resolved
        assert "$ref" not in str(result)
        # Nested structure should be correct
        assert result["properties"]["team"]["properties"]["leader"]["properties"]["name"]["type"] == "string"


class TestLoadModelConfig:
    """Tests for model configuration loading."""
    
    @pytest.fixture
    def runtime_env(self):
        """Create a namespace with the runtime loaded."""
        env = {}
        exec(get_preamble(), env)
        return env
    
    def test_unknown_model_returns_defaults(self, runtime_env):
        """Unknown model should return default OpenAI config."""
        load_model_config = runtime_env['load_model_config']
        
        config = load_model_config("nonexistent-model-xyz")
        
        assert config["type"] == "openai"
        assert config["cost_in"] == 0
        assert config["cost_out"] == 0
    
    def test_loads_from_models_json(self, runtime_env):
        """Should load config from models.json if it exists."""
        load_model_config = runtime_env['load_model_config']
        os_module = runtime_env['os']
        
        # Create a temporary models.json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "test-model": {
                    "type": "gemini",
                    "cost_in": 1.5,
                    "cost_out": 3.0
                }
            }, f)
            temp_path = f.name
        
        # Temporarily change to temp directory
        original_cwd = os.getcwd()
        temp_dir = os.path.dirname(temp_path)
        os.chdir(temp_dir)
        os.rename(temp_path, os.path.join(temp_dir, "models.json"))
        
        try:
            config = load_model_config("test-model")
            assert config["type"] == "gemini"
            assert config["cost_in"] == 1.5
            assert config["cost_out"] == 3.0
        finally:
            os.chdir(original_cwd)
            try:
                os.unlink(os.path.join(temp_dir, "models.json"))
            except:
                pass
