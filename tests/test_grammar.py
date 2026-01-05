"""
Unit tests for the Ensō grammar.
"""
import pytest
from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF
from core.grammar import enso_grammar


@pytest.fixture
def parser():
    """Create a parser instance for testing."""
    return Lark(enso_grammar, parser='earley')


class TestStructParsing:
    """Tests for struct definition parsing."""
    
    def test_simple_struct(self, parser):
        """Parse a simple struct with basic types."""
        code = 'struct Person { name: String, age: Int }'
        tree = parser.parse(code)
        assert tree is not None
    
    def test_struct_with_list_type(self, parser):
        """Parse a struct with List<T> type."""
        code = 'struct Team { members: List<String> }'
        tree = parser.parse(code)
        assert tree is not None
    
    def test_struct_with_enum_type(self, parser):
        """Parse a struct with Enum type."""
        code = 'struct Task { status: Enum<"pending", "done"> }'
        tree = parser.parse(code)
        assert tree is not None
    
    def test_empty_struct(self, parser):
        """Parse an empty struct."""
        code = 'struct Empty { }'
        tree = parser.parse(code)
        assert tree is not None


class TestAIFunctionParsing:
    """Tests for AI function definition parsing."""
    
    def test_minimal_ai_function(self, parser):
        """Parse a minimal AI function with required fields."""
        code = '''
        ai fn classify(text: String) -> Label {
            instruction: "Classify the text",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_ai_function_with_system_instruction(self, parser):
        """Parse an AI function with system_instruction."""
        code = '''
        ai fn analyze(text: String) -> Analysis {
            system_instruction: "You are an expert analyzer",
            instruction: "Analyze the text",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_ai_function_with_temperature(self, parser):
        """Parse an AI function with temperature config."""
        code = '''
        ai fn creative(prompt: String) -> Story {
            instruction: "Write a story",
            model: "gpt-4o",
            temperature: 0.9
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_ai_function_with_result_type(self, parser):
        """Parse an AI function returning Result<T, E>."""
        code = '''
        ai fn safe_classify(text: String) -> Result<Label, Error> {
            instruction: "Classify safely",
            model: "gpt-4o"
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_ai_function_with_examples(self, parser):
        """Parse an AI function with few-shot examples."""
        code = '''
        ai fn sentiment(text: String) -> Sentiment {
            instruction: "Analyze sentiment",
            model: "gpt-4o",
            examples: [
                (input: "I love this!", expected: "positive"),
                (input: "I hate this!", expected: "negative"),
            ]
        }
        '''
        tree = parser.parse(code)
        assert tree is not None


class TestRegularFunctionParsing:
    """Tests for regular function definition parsing."""
    
    def test_simple_function(self, parser):
        """Parse a simple regular function."""
        code = '''
        fn add(a: Int, b: Int) -> Int {
            return a + b;
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_function_with_if_else(self, parser):
        """Parse a function with if/else statement."""
        code = '''
        fn max(a: Int, b: Int) -> Int {
            if a > b {
                return a;
            } else {
                return b;
            }
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_function_with_for_loop(self, parser):
        """Parse a function with for loop."""
        code = '''
        fn process(items: List<String>) -> Int {
            let count = 0;
            for item in items {
                print(item);
            }
            return count;
        }
        '''
        tree = parser.parse(code)
        assert tree is not None


class TestTestBlockParsing:
    """Tests for test block parsing."""
    
    def test_simple_test(self, parser):
        """Parse a simple test block."""
        code = '''
        test "basic test" {
            let x = 5;
            assert x == 5;
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_ai_test(self, parser):
        """Parse an AI test block."""
        code = '''
        test ai "real api test" {
            let result = classify("hello");
            assert result.value != "";
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_test_with_mock(self, parser):
        """Parse a test block with mock statement."""
        code = '''
        test "mocked test" {
            mock classify => Label { category: "test" };
            let result = classify("hello");
            assert result.value.category == "test";
        }
        '''
        tree = parser.parse(code)
        assert tree is not None


class TestMatchParsing:
    """Tests for match expression parsing."""
    
    def test_match_ok_err(self, parser):
        """Parse a match expression with Ok and Err arms."""
        code = '''
        fn handle(result: Result) -> String {
            match result {
                Ok(value) => {
                    return value;
                },
                Err(error) => {
                    return "error";
                },
            }
        }
        '''
        tree = parser.parse(code)
        assert tree is not None


class TestExpressionParsing:
    """Tests for expression parsing."""
    
    def test_property_access(self, parser):
        """Parse property access expressions."""
        code = '''
        fn get_name(person: Person) -> String {
            return person.name;
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_function_call(self, parser):
        """Parse function call expressions."""
        code = '''
        fn main() {
            let result = process("input");
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_struct_initialization(self, parser):
        """Parse struct initialization expressions."""
        code = '''
        fn create() -> Person {
            return Person { name: "Alice", age: 30 };
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_list_literal(self, parser):
        """Parse list literal expressions."""
        code = '''
        fn get_items() -> List<String> {
            return ["a", "b", "c"];
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_comparison_operators(self, parser):
        """Parse comparison operator expressions."""
        code = '''
        fn check(a: Int, b: Int) -> Int {
            if a == b {
                return 0;
            }
            if a != b {
                return 1;
            }
            if a >= b {
                return 2;
            }
            if a <= b {
                return 3;
            }
            return 4;
        }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_logical_operators(self, parser):
        """Parse logical operator expressions."""
        code = '''
        fn check(a: Int, b: Int) -> Int {
            if a > 0 && b > 0 {
                return 1;
            }
            if a > 0 || b > 0 {
                return 2;
            }
            return 0;
        }
        '''
        tree = parser.parse(code)
        assert tree is not None


class TestCommentParsing:
    """Tests for comment handling."""
    
    def test_single_line_comment_slash(self, parser):
        """Parse code with // comments."""
        code = '''
        // This is a comment
        struct Foo { }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_single_line_comment_hash(self, parser):
        """Parse code with # comments."""
        code = '''
        # This is a comment
        struct Foo { }
        '''
        tree = parser.parse(code)
        assert tree is not None
    
    def test_block_comment(self, parser):
        """Parse code with block comments."""
        code = '''
        /* This is a
           multi-line comment */
        struct Foo { }
        '''
        tree = parser.parse(code)
        assert tree is not None


class TestInvalidSyntax:
    """Tests for invalid syntax detection."""
    
    def test_missing_semicolon(self, parser):
        """Missing semicolon should cause parse error."""
        code = 'let x = 5'  # Missing semicolon
        with pytest.raises((UnexpectedCharacters, UnexpectedToken, UnexpectedEOF)):
            parser.parse(code)
    
    def test_missing_brace(self, parser):
        """Missing brace should cause parse error."""
        code = 'struct Foo { name: String'  # Missing closing brace
        with pytest.raises((UnexpectedCharacters, UnexpectedToken, UnexpectedEOF)):
            parser.parse(code)
