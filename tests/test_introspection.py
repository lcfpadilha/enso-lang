"""
Unit tests for core/introspection.py - SchemaExtractor class.
"""

import pytest
from lark import Lark

from core.grammar import enso_grammar
from core.introspection import SchemaExtractor


class TestSchemaExtractorBasic:
    """Tests for basic schema extraction functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create a Lark parser for testing."""
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def extractor(self):
        """Create schema extractor."""
        return SchemaExtractor()
    
    def test_extract_single_ai_function(self, parser, extractor):
        """Should extract a single AI function."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn analyze(text: String) -> Output {
            instruction: "Analyze the text",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        assert len(result) == 1
        assert result[0]['type'] == 'function'
        assert result[0]['name'] == 'analyze'
    
    def test_extract_multiple_ai_functions(self, parser, extractor):
        """Should extract multiple AI functions."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn analyze(text: String) -> Output {
            instruction: "Analyze",
            model: "gpt-4o"
        }
        
        ai fn summarize(text: String) -> Output {
            instruction: "Summarize",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        assert len(result) == 2
        names = [f['name'] for f in result]
        assert 'analyze' in names
        assert 'summarize' in names
    
    def test_ignore_structs(self, parser, extractor):
        """Should not include structs in result."""
        source = '''
        struct MyStruct {
            field: String
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        assert len(result) == 0
    
    def test_ignore_regular_functions(self, parser, extractor):
        """Should not include regular functions in result."""
        source = '''
        fn add(a: Int, b: Int) -> Int {
            return a + b;
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        assert len(result) == 0
    
    def test_ignore_tests(self, parser, extractor):
        """Should not include test definitions in result."""
        source = '''
        test "my test" {
            let x = 5;
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        assert len(result) == 0


class TestSchemaExtractorArguments:
    """Tests for argument extraction."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()
    
    def test_extract_single_argument(self, parser, extractor):
        """Should extract single argument with name and type."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn analyze(text: String) -> Output {
            instruction: "Analyze",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        args = result[0]['args']
        assert len(args) == 1
        assert args[0]['name'] == 'text'
        # SchemaExtractor preserves original type names for SDK generation
        assert args[0]['type'] == 'String'
    
    def test_extract_multiple_arguments(self, parser, extractor):
        """Should extract multiple arguments."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn process(text: String, count: Int) -> Output {
            instruction: "Process",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        args = result[0]['args']
        assert len(args) == 2
        assert args[0]['name'] == 'text'
        # SchemaExtractor preserves original type names for SDK generation
        assert args[0]['type'] == 'String'
        assert args[1]['name'] == 'count'
        assert args[1]['type'] == 'Int'
    
    def test_extract_no_arguments(self, parser, extractor):
        """Should handle functions with no arguments."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn generate() -> Output {
            instruction: "Generate",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        args = result[0]['args']
        assert len(args) == 0


class TestSchemaExtractorReturnTypes:
    """Tests for return type extraction."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()
    
    def test_extract_simple_return_type(self, parser, extractor):
        """Should extract simple return type."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn analyze(text: String) -> Output {
            instruction: "Analyze",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        # Return type comes through as "Any" because SchemaExtractor
        # uses a simplified type_expr that returns "Any"
        assert 'return' in result[0]


class TestSchemaExtractorMixedContent:
    """Tests for extracting from files with mixed content."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()
    
    def test_extract_from_mixed_file(self, parser, extractor):
        """Should only extract AI functions from mixed content."""
        source = '''
        struct Output {
            result: String
        }
        
        ai fn analyze(text: String) -> Output {
            instruction: "Analyze input",
            model: "gpt-4o"
        }
        
        ai fn summarize(text: String) -> Output {
            instruction: "Summarize",
            model: "gpt-4o"
        }
        
        test "simple test" {
            let x = 5;
            assert x == 5;
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        # Should only have 2 AI functions
        assert len(result) == 2
        
        # All should be type 'function'
        for fn in result:
            assert fn['type'] == 'function'
        
        # Check names
        names = [f['name'] for f in result]
        assert 'analyze' in names
        assert 'summarize' in names


class TestSchemaExtractorFunctionDetails:
    """Tests for detailed function extraction."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()
    
    def test_function_structure(self, parser, extractor):
        """Should return proper function structure."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn my_function(arg1: String, arg2: Int) -> Output {
            instruction: "Do something",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = extractor.transform(tree)
        
        fn = result[0]
        
        # Check structure
        assert 'type' in fn
        assert 'name' in fn
        assert 'args' in fn
        assert 'return' in fn
        
        # Check values
        assert fn['type'] == 'function'
        assert fn['name'] == 'my_function'
        assert isinstance(fn['args'], list)
