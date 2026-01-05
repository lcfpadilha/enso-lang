"""
Unit tests for core/transformer.py - EnsoTransformer class.
"""

import pytest
from lark import Lark
from lark.exceptions import VisitError

from core.grammar import enso_grammar
from core.transformer import EnsoTransformer


class TestEnsoTransformerBasic:
    """Tests for basic transformation functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create a Lark parser for testing."""
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def transformer(self):
        """Create transformer with minimal runtime preamble for testing."""
        # Use empty preamble to make output easier to test
        return EnsoTransformer(runtime_preamble="# Runtime\n")
    
    def test_struct_transformation(self, parser, transformer):
        """Struct should transform to Pydantic BaseModel."""
        source = """
        struct Person {
            name: String,
            age: Int
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "class Person(BaseModel):" in result
        assert "name: str" in result
        assert "age: int" in result
    
    def test_ai_function_transformation(self, parser, transformer):
        """AI function should transform to Python function with EnsoAgent."""
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
        result = transformer.transform(tree)
        
        assert "def analyze(text: str):" in result
        assert "EnsoAgent(" in result
        assert 'name="analyze"' in result
        assert 'instruction="Analyze the text"' in result
        assert 'model="gpt-4o"' in result
    
    def test_regular_function_transformation(self, parser, transformer):
        """Regular function should transform to Python function."""
        source = """
        fn add(a: Int, b: Int) -> Int {
            return a + b;
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "def add(a: int, b: int):" in result
        assert "return a + b" in result
    
    def test_test_block_transformation(self, parser, transformer):
        """Test block should transform to test function."""
        source = '''
        test "my test" {
            let x = 5;
            assert x == 5;
        }
        '''
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "def test_my_test():" in result
        assert "x = 5" in result
        assert "assert x == 5" in result
    
    def test_ai_test_block_transformation(self, parser, transformer):
        """AI test block should have AI_ prefix."""
        source = '''
        test ai "real api test" {
            let x = 1;
        }
        '''
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "def test_AI_real_api_test():" in result


class TestEnsoTransformerStatements:
    """Tests for statement transformations."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def transformer(self):
        return EnsoTransformer(runtime_preamble="# Runtime\n")
    
    def test_let_statement(self, parser, transformer):
        """Let statement should become Python assignment."""
        source = """
        fn main() -> Int {
            let x = 5;
            return x;
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "x = 5" in result
    
    def test_if_statement(self, parser, transformer):
        """If statement should transform correctly."""
        source = """
        fn check(x: Int) -> Int {
            if x > 0 {
                return 1;
            } else {
                return 0;
            }
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "if x > 0:" in result
        assert "return 1" in result
        assert "else:" in result
        assert "return 0" in result
    
    def test_for_loop(self, parser, transformer):
        """For loop should transform correctly."""
        source = """
        fn process(items: List<Int>) -> Int {
            for item in items {
                print(item);
            }
            return 0;
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "for item in items:" in result
        assert "print(item)" in result
    
    def test_match_statement(self, parser, transformer):
        """Match statement should transform to if/elif chain."""
        source = """
        struct Data {
            value: Int
        }
        
        fn handle(result: Result<Data, String>) -> Int {
            match result {
                Ok(data) => {
                    return data.value;
                },
                Err(e) => {
                    return 0;
                }
            }
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "if isinstance(result, Ok):" in result
        assert "data = result.value" in result
        assert "elif isinstance(result, Err):" in result
        assert "e = result.error" in result


class TestEnsoTransformerTypes:
    """Tests for type transformations."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def transformer(self):
        return EnsoTransformer(runtime_preamble="# Runtime\n")
    
    def test_simple_types(self, parser, transformer):
        """Simple types should map to Python types."""
        source = """
        struct Types {
            s: String,
            i: Int,
            f: Float
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "s: str" in result
        assert "i: int" in result
        assert "f: float" in result
    
    def test_list_type(self, parser, transformer):
        """List<T> should transform to List[T]."""
        source = """
        struct Container {
            items: List<String>
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "items: List[str]" in result
    
    def test_enum_type(self, parser, transformer):
        """Enum<...> should transform to Literal[...]."""
        source = """
        struct Status {
            state: Enum<"active", "inactive">
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert 'state: Literal["active", "inactive"]' in result


class TestEnsoTransformerExpressions:
    """Tests for expression transformations."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def transformer(self):
        return EnsoTransformer(runtime_preamble="# Runtime\n")
    
    def test_struct_initialization(self, parser, transformer):
        """Struct initialization should transform correctly."""
        source = """
        struct Point {
            x: Int,
            y: Int
        }
        
        fn make_point() -> Point {
            return Point { x: 1, y: 2 };
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "return Point(x=1, y=2)" in result
    
    def test_list_literal(self, parser, transformer):
        """List literal should transform correctly."""
        source = """
        fn get_list() -> List<Int> {
            return [1, 2, 3];
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "return [1, 2, 3]" in result
    
    def test_property_access(self, parser, transformer):
        """Property access should use dot notation."""
        source = """
        struct Person {
            name: String
        }
        
        fn get_name(p: Person) -> String {
            return p.name;
        }
        """
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "return p.name" in result


class TestEnsoTransformerMocking:
    """Tests for mock statement transformation."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def transformer(self):
        return EnsoTransformer(runtime_preamble="# Runtime\n")
    
    def test_mock_statement(self, parser, transformer):
        """Mock statement should set MOCKS dict."""
        source = '''
        struct Output {
            value: Int
        }
        
        test "with mock" {
            mock my_func => Output { value: 42 };
            let result = my_func("input");
        }
        '''
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "MOCKS['my_func'] = Output(value=42)" in result


class TestEnsoTransformerValidation:
    """Tests for validation during transformation."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def transformer(self):
        return EnsoTransformer(runtime_preamble="# Runtime\n")
    
    def test_undefined_variable_in_instruction(self, parser, transformer):
        """Should raise error for undefined variables in instruction."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn analyze(text: String) -> Output {
            instruction: "Process {undefined_var}",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        
        with pytest.raises(VisitError) as exc_info:
            transformer.transform(tree)
        
        # Check the underlying error message
        assert "Undefined variables" in str(exc_info.value)
        assert "undefined_var" in str(exc_info.value)
    
    def test_valid_variable_in_instruction(self, parser, transformer):
        """Should accept valid variables in instruction."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn analyze(text: String) -> Output {
            instruction: "Process {text}",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        # Should include f-string conversion
        assert 'f"Process {text}"' in result


class TestEnsoTransformerAIFeatures:
    """Tests for AI-specific features."""
    
    @pytest.fixture
    def parser(self):
        return Lark(enso_grammar, parser='earley')
    
    @pytest.fixture
    def transformer(self):
        return EnsoTransformer(runtime_preamble="# Runtime\n")
    
    def test_system_instruction(self, parser, transformer):
        """System instruction should be included in agent args."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn analyze(text: String) -> Output {
            system_instruction: "You are a helpful assistant",
            instruction: "Analyze this",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert 'system_instruction="You are a helpful assistant"' in result
    
    def test_temperature_setting(self, parser, transformer):
        """Temperature should be included in agent args."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn creative(prompt: String) -> Output {
            instruction: "Be creative",
            model: "gpt-4o",
            temperature: 0.9
        }
        '''
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        assert "temperature=0.9" in result
    
    def test_result_type_extraction(self, parser, transformer):
        """Result<T, E> should extract T for response model."""
        source = '''
        struct Output {
            value: String
        }
        
        ai fn analyze(text: String) -> Result<Output, String> {
            instruction: "Analyze",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(source)
        result = transformer.transform(tree)
        
        # Response model should be Output, not the full Result type
        assert "analyze_response_model = Output" in result
