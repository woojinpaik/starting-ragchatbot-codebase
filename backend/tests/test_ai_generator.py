import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator tool calling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-3-haiku-20240307"

        # Mock anthropic client
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)

    def test_generate_response_without_tools(self):
        """Test basic response generation without tools"""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a test response")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Generate response
        result = self.ai_generator.generate_response("What is Python?")

        # Verify API call
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args
        self.assertEqual(call_args[1]["model"], self.model)
        self.assertEqual(call_args[1]["messages"][0]["content"], "What is Python?")
        self.assertNotIn("tools", call_args[1])

        # Verify result
        self.assertEqual(result, "This is a test response")

    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        conversation_history = "User: Hello\nAssistant: Hi there!"

        result = self.ai_generator.generate_response(
            "Follow up question", conversation_history=conversation_history
        )

        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        self.assertIn("Previous conversation:", system_content)
        self.assertIn("User: Hello\nAssistant: Hi there!", system_content)

        self.assertEqual(result, "Response with history")

    def test_generate_response_with_tools_no_tool_use(self):
        """Test response generation with tools available but no tool use"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct answer without tools")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        result = self.ai_generator.generate_response("What is 2+2?", tools=tools)

        # Verify tools were provided to API
        call_args = self.mock_client.messages.create.call_args
        self.assertEqual(call_args[1]["tools"], tools)
        self.assertEqual(call_args[1]["tool_choice"], {"type": "auto"})

        self.assertEqual(result, "Direct answer without tools")

    def test_generate_response_with_tool_use(self):
        """Test response generation with tool execution"""
        # Mock initial response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "python basics"}
        mock_tool_content.id = "tool_use_123"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        mock_initial_response.stop_reason = "tool_use"

        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Based on search results: Python is a programming language")
        ]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python is a high-level programming language"

        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        result = self.ai_generator.generate_response(
            "What is Python?", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="python basics"
        )

        # Verify two API calls were made
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        # Verify final response
        self.assertEqual(result, "Based on search results: Python is a programming language")

    def test_tool_execution_with_multiple_tools(self):
        """Test handling multiple tool calls in one response"""
        # Mock response with multiple tool uses
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "search_course_content"
        mock_tool_content_1.input = {"query": "python"}
        mock_tool_content_1.id = "tool_use_1"

        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "get_course_outline"
        mock_tool_content_2.input = {"course_name": "Python Basics"}
        mock_tool_content_2.id = "tool_use_2"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content_1, mock_tool_content_2]
        mock_initial_response.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Combined response from multiple tools")]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result content",
            "Course outline content",
        ]

        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            },
            {
                "name": "get_course_outline",
                "description": "Outline",
                "input_schema": {},
            },
        ]

        result = self.ai_generator.generate_response(
            "Tell me about Python course", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="python")
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_name="Python Basics"
        )

        # Verify final API call structure
        final_call_args = self.mock_client.messages.create.call_args_list[1]
        messages = final_call_args[1]["messages"]

        # Should have: original user message, assistant tool use, user tool results
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")

        # Tool results should be structured correctly
        tool_results = messages[2]["content"]
        self.assertEqual(len(tool_results), 2)
        self.assertEqual(tool_results[0]["type"], "tool_result")
        self.assertEqual(tool_results[0]["tool_use_id"], "tool_use_1")
        self.assertEqual(tool_results[0]["content"], "Search result content")

        self.assertEqual(result, "Combined response from multiple tools")

    def test_tool_execution_error_handling(self):
        """Test handling of tool execution when tool manager fails"""
        # Mock response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_use_123"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        mock_initial_response.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Error handled gracefully")]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Mock tool manager that returns error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool 'search_course_content' not found"

        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]

        result = self.ai_generator.generate_response(
            "Test query", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool execution was attempted
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test"
        )

        # Verify error message was passed to final API call
        final_call_args = self.mock_client.messages.create.call_args_list[1]
        tool_results = final_call_args[1]["messages"][2]["content"]
        self.assertEqual(tool_results[0]["content"], "Tool 'search_course_content' not found")

        self.assertEqual(result, "Error handled gracefully")

    def test_system_prompt_content(self):
        """Test that system prompt contains expected tool usage guidelines"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        self.ai_generator.generate_response("Test query")

        call_args = self.mock_client.messages.create.call_args
        system_content = call_args[1]["system"]

        # Verify key instructions are present
        self.assertIn("search_course_content", system_content)
        self.assertIn("get_course_outline", system_content)
        self.assertIn("Tool Usage Guidelines", system_content)
        self.assertIn("Multi-round tool calling", system_content)
        self.assertIn("Maximum 2 rounds of tool usage per query", system_content)
        self.assertIn("No meta-commentary", system_content)

    def test_api_parameters_configuration(self):
        """Test that API parameters are set correctly"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        self.ai_generator.generate_response("Test query")

        call_args = self.mock_client.messages.create.call_args
        params = call_args[1]

        # Verify base parameters
        self.assertEqual(params["model"], self.model)
        self.assertEqual(params["temperature"], 0)
        self.assertEqual(params["max_tokens"], 800)
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Test query")


class TestAIGeneratorToolIntegration(unittest.TestCase):
    """Integration tests for AI generator with actual tool definitions"""

    def setUp(self):
        """Set up test fixtures with realistic tool definitions"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator("test_key", "test_model")

        # Realistic tool definitions
        self.tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the course content",
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work)",
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_course_outline",
                "description": "Get course outline with title, link, and complete lesson list",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work)",
                        }
                    },
                    "required": ["course_name"],
                },
            },
        ]

    def test_realistic_search_query_flow(self):
        """Test realistic flow for content search query"""
        # Mock tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {
            "query": "variables in python",
            "course_name": "Python Basics",
        }
        mock_tool_content.id = "tool_use_123"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        mock_initial_response.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="Variables in Python are used to store data. They are created when you assign a value to them."
            )
        ]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "[Python Basics - Lesson 2]\nVariables are containers for storing data values. In Python, you create a variable the moment you first assign a value to it."

        result = self.ai_generator.generate_response(
            "How do variables work in Python basics course?",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify tool was called with appropriate parameters
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="variables in python",
            course_name="Python Basics",
        )

        self.assertIn("Variables in Python are used to store data", result)

    def test_realistic_outline_query_flow(self):
        """Test realistic flow for course outline query"""
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "get_course_outline"
        mock_tool_content.input = {"course_name": "Python Basics"}
        mock_tool_content.id = "tool_use_456"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        mock_initial_response.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="The Python Basics course covers fundamental programming concepts with 5 lessons including variables, functions, and control structures."
            )
        ]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "**Python Basics**\nCourse Link: http://example.com/python\n\n**Lessons:**\nLesson 1: Introduction\nLesson 2: Variables\nLesson 3: Functions\nLesson 4: Control Structures\nLesson 5: Practice"

        result = self.ai_generator.generate_response(
            "What is the structure of the Python Basics course?",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "get_course_outline", course_name="Python Basics"
        )

        self.assertIn("Python Basics course covers fundamental programming concepts", result)


class TestAIGeneratorMultiRound(unittest.TestCase):
    """Test cases for multi-round tool calling functionality"""

    def setUp(self):
        """Set up test fixtures for multi-round testing"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator("test_key", "test_model")

        self.tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_course_outline",
                "description": "Get course outline",
                "input_schema": {
                    "type": "object",
                    "properties": {"course_name": {"type": "string"}},
                    "required": ["course_name"],
                },
            },
        ]

    def test_two_round_tool_calling_success(self):
        """Test successful 2-round tool calling flow"""
        # Mock first round response with tool use
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "get_course_outline"
        mock_tool_content_1.input = {"course_name": "Course X"}
        mock_tool_content_1.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content_1]
        mock_response_1.stop_reason = "tool_use"

        # Mock second round response with different tool use
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {"query": "lesson 4 topic"}
        mock_tool_content_2.id = "tool_use_2"

        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_content_2]
        mock_response_2.stop_reason = "tool_use"

        # Mock final response after second tool execution
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="Based on the course outline and content search, lesson 4 covers advanced concepts."
            )
        ]
        mock_final_response.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course X outline: Lesson 1: Intro, Lesson 2: Basics, Lesson 3: Intermediate, Lesson 4: Advanced Topics",
            "Lesson 4 content: Advanced concepts include machine learning, neural networks, and AI applications",
        ]

        result = self.ai_generator.generate_response(
            "What does lesson 4 of Course X cover?",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify 3 API calls were made (2 rounds + final)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Course X")
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="lesson 4 topic"
        )

        # Verify final response
        self.assertIn("lesson 4 covers advanced concepts", result)

    def test_single_round_with_no_follow_up_needed(self):
        """Test that single round works when no follow-up tool calls are needed"""
        # Mock response with tool use but Claude doesn't need more tools
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "python basics"}
        mock_tool_content.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content]
        mock_response_1.stop_reason = "tool_use"

        # Mock second round response with direct answer (no tool use)
        mock_response_2 = Mock()
        mock_response_2.content = [
            Mock(
                text="Python is a programming language used for web development, data science, and automation."
            )
        ]
        mock_response_2.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python is a high-level programming language"

        result = self.ai_generator.generate_response(
            "What is Python?", tools=self.tools, tool_manager=mock_tool_manager
        )

        # Verify 2 API calls (1 round + response after tool execution)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        # Verify only one tool was executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)

        self.assertIn("programming language", result)

    def test_max_rounds_termination(self):
        """Test termination when max rounds (2) is reached"""
        # Mock both rounds with tool use
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "get_course_outline"
        mock_tool_content_1.input = {"course_name": "Course A"}
        mock_tool_content_1.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content_1]
        mock_response_1.stop_reason = "tool_use"

        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {"query": "comparison topic"}
        mock_tool_content_2.id = "tool_use_2"

        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_content_2]
        mock_response_2.stop_reason = "tool_use"

        # Mock final response when max rounds reached
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Based on available information, here's the comparison.")
        ]
        mock_final_response.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course A outline: Lesson 1, Lesson 2, Lesson 3",
            "Comparison content found in Course A materials",
        ]

        result = self.ai_generator.generate_response(
            "Compare Course A with others",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify 3 API calls (max 2 rounds + final call without tools)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify final call was made without tools
        final_call_args = self.mock_client.messages.create.call_args_list[2]
        self.assertNotIn("tools", final_call_args[1])

        self.assertIn("comparison", result)

    def test_tool_error_in_first_round(self):
        """Test error handling when first round tool execution fails"""
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test query"}
        mock_tool_content.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content]
        mock_response_1.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="I apologize, but I encountered an error while searching. Let me provide a general answer."
            )
        ]
        mock_final_response.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_final_response,
        ]

        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool not found")

        result = self.ai_generator.generate_response(
            "Search for something", tools=self.tools, tool_manager=mock_tool_manager
        )

        # Verify 2 API calls (1 failed round + final)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        # Verify tool execution was attempted
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)

        # Verify final response contains error handling
        self.assertIn("error", result.lower())

    def test_conversation_context_preserved_across_rounds(self):
        """Test that conversation context builds correctly across rounds"""
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "get_course_outline"
        mock_tool_content.input = {"course_name": "Python Course"}
        mock_tool_content.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content]
        mock_response_1.stop_reason = "tool_use"

        mock_response_2 = Mock()
        mock_response_2.content = [Mock(text="Final comprehensive answer")]
        mock_response_2.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Course outline content"

        self.ai_generator.generate_response(
            "Tell me about Python course structure",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify second API call has the complete conversation context
        second_call_args = self.mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]

        # Should have: original user message, assistant tool use, user tool results
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")

        # Tool results should be in the third message
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")
        self.assertEqual(messages[2]["content"][0]["content"], "Course outline content")


class TestAIGeneratorComparisonQueries(unittest.TestCase):
    """Test cases for complex comparison queries requiring multiple searches"""

    def setUp(self):
        """Set up test fixtures for comparison query testing"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator("test_key", "test_model")

        self.tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "course_name": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_course_outline",
                "description": "Get course outline",
                "input_schema": {
                    "type": "object",
                    "properties": {"course_name": {"type": "string"}},
                    "required": ["course_name"],
                },
            },
        ]

    def test_comparison_query_across_courses(self):
        """Test comparison query that requires searching multiple courses"""
        # Round 1: Search for topic in first course
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "search_course_content"
        mock_tool_content_1.input = {
            "query": "machine learning",
            "course_name": "AI Fundamentals",
        }
        mock_tool_content_1.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content_1]
        mock_response_1.stop_reason = "tool_use"

        # Round 2: Search for same topic in different course
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {
            "query": "machine learning",
            "course_name": "Data Science",
        }
        mock_tool_content_2.id = "tool_use_2"

        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_content_2]
        mock_response_2.stop_reason = "tool_use"

        # Final response with comparison
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="AI Fundamentals focuses on theoretical ML concepts, while Data Science emphasizes practical implementation."
            )
        ]
        mock_final_response.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "AI Fundamentals ML content: Theoretical foundations, algorithms, neural networks theory",
            "Data Science ML content: Practical implementation, scikit-learn, model deployment",
        ]

        result = self.ai_generator.generate_response(
            "Compare how machine learning is taught in AI Fundamentals vs Data Science courses",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify both searches were performed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="machine learning",
            course_name="AI Fundamentals",
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="machine learning",
            course_name="Data Science",
        )

        # Verify comparison in result
        self.assertIn("theoretical", result.lower())
        self.assertIn("practical", result.lower())

    def test_multi_part_question_requiring_outline_then_content(self):
        """Test multi-part question requiring course outline then specific content search"""
        # Round 1: Get course outline to understand structure
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "get_course_outline"
        mock_tool_content_1.input = {"course_name": "Web Development"}
        mock_tool_content_1.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content_1]
        mock_response_1.stop_reason = "tool_use"

        # Round 2: Search specific lesson content based on outline
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {
            "query": "lesson 3 frameworks",
            "course_name": "Web Development",
        }
        mock_tool_content_2.id = "tool_use_2"

        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_content_2]
        mock_response_2.stop_reason = "tool_use"

        # Final comprehensive answer
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="The Web Development course has 5 lessons. Lesson 3 focuses on frameworks like React and Vue."
            )
        ]
        mock_final_response.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Web Development Course:\nLesson 1: HTML Basics\nLesson 2: CSS Styling\nLesson 3: JavaScript Frameworks\nLesson 4: Backend Integration\nLesson 5: Deployment",
            "Lesson 3 content: Introduction to modern frameworks including React, Vue, and Angular. Covers component architecture and state management.",
        ]

        result = self.ai_generator.generate_response(
            "What is the structure of Web Development course and what frameworks are covered?",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify both tools were used in sequence
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_name="Web Development"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="lesson 3 frameworks",
            course_name="Web Development",
        )

        # Verify comprehensive answer
        self.assertIn("5 lessons", result)
        self.assertIn("React", result)

    def test_cross_lesson_analysis_query(self):
        """Test query requiring analysis across different lessons"""
        # Round 1: Search content from one lesson
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "search_course_content"
        mock_tool_content_1.input = {
            "query": "lesson 2 variables",
            "course_name": "Programming Basics",
        }
        mock_tool_content_1.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content_1]
        mock_response_1.stop_reason = "tool_use"

        # Round 2: Search related content from different lesson
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {
            "query": "lesson 4 functions",
            "course_name": "Programming Basics",
        }
        mock_tool_content_2.id = "tool_use_2"

        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_content_2]
        mock_response_2.stop_reason = "tool_use"

        # Final analysis response
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="Variables from lesson 2 are used as parameters and return values in functions covered in lesson 4."
            )
        ]
        mock_final_response.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Lesson 2 Variables: Variables store data values. They can be strings, numbers, or booleans.",
            "Lesson 4 Functions: Functions take parameters (variables) and can return values (also variables).",
        ]

        result = self.ai_generator.generate_response(
            "How do variables from lesson 2 relate to functions in lesson 4?",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify searches for both lessons
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="lesson 2 variables",
            course_name="Programming Basics",
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="lesson 4 functions",
            course_name="Programming Basics",
        )

        # Verify analytical response
        self.assertIn("parameters", result)
        self.assertIn("return values", result)


class TestAIGeneratorTerminationAndErrorHandling(unittest.TestCase):
    """Test cases for termination conditions and error handling in multi-round tool calling"""

    def setUp(self):
        """Set up test fixtures for termination and error testing"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator("test_key", "test_model")

        self.tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

    def test_early_termination_no_tools_needed(self):
        """Test early termination when Claude doesn't need any tools"""
        # Mock direct response without tool use
        mock_response = Mock()
        mock_response.content = [
            Mock(text="This is a general knowledge answer that doesn't require tools.")
        ]
        mock_response.stop_reason = "end_turn"

        self.mock_client.messages.create.return_value = mock_response

        mock_tool_manager = Mock()

        result = self.ai_generator.generate_response(
            "What is 2 + 2?", tools=self.tools, tool_manager=mock_tool_manager
        )

        # Verify only one API call was made
        self.assertEqual(self.mock_client.messages.create.call_count, 1)

        # Verify no tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 0)

        # Verify tools were available in the first call
        call_args = self.mock_client.messages.create.call_args
        self.assertIn("tools", call_args[1])

        self.assertEqual(result, "This is a general knowledge answer that doesn't require tools.")

    def test_termination_after_tool_execution_failure(self):
        """Test termination when tool execution fails completely"""
        # Mock response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content]
        mock_response_1.stop_reason = "tool_use"

        # Mock final response after tool failure
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="I apologize, but I encountered an error accessing the course materials.")
        ]
        mock_final_response.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_final_response,
        ]

        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        result = self.ai_generator.generate_response(
            "Search for something", tools=self.tools, tool_manager=mock_tool_manager
        )

        # Verify 2 API calls (failed tool round + final response)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        # Verify tool execution was attempted once
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)

        # Verify second call includes error in conversation context
        second_call_args = self.mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]

        # Should have user query, assistant tool use, user tool error result
        self.assertEqual(len(messages), 3)
        tool_result_content = messages[2]["content"][0]["content"]
        self.assertIn("Tool execution error", tool_result_content)

        self.assertIn("error", result.lower())

    def test_no_tool_manager_provided(self):
        """Test behavior when tools are available but no tool manager is provided"""
        # Mock response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_use_1"

        mock_response = Mock()
        mock_response.content = [mock_tool_content]
        mock_response.stop_reason = "tool_use"

        self.mock_client.messages.create.return_value = mock_response

        result = self.ai_generator.generate_response(
            "Search for something",
            tools=self.tools,
            tool_manager=None,  # No tool manager provided
        )

        # Verify only one API call was made
        self.assertEqual(self.mock_client.messages.create.call_count, 1)

        # Should return empty string when no tool manager but tools needed
        self.assertEqual(result, "")

    def test_api_error_during_first_round(self):
        """Test handling of API errors during tool calling"""
        # Mock API exception
        self.mock_client.messages.create.side_effect = Exception("API rate limit exceeded")

        mock_tool_manager = Mock()

        # Should propagate the exception
        with self.assertRaises(Exception) as context:
            self.ai_generator.generate_response(
                "Test query", tools=self.tools, tool_manager=mock_tool_manager
            )

        self.assertIn("API rate limit exceeded", str(context.exception))

    def test_termination_at_second_round_limit(self):
        """Test termination when reaching the 2-round limit with pending tool use"""
        # Mock first round with tool use
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "search_course_content"
        mock_tool_content_1.input = {"query": "first search"}
        mock_tool_content_1.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content_1]
        mock_response_1.stop_reason = "tool_use"

        # Mock second round also with tool use (should trigger final call)
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {"query": "second search"}
        mock_tool_content_2.id = "tool_use_2"

        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_content_2]
        mock_response_2.stop_reason = "tool_use"

        # Mock final call without tools
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Based on available information from searches, here's my answer.")
        ]
        mock_final_response.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First search results",
            "Second search results",
        ]

        result = self.ai_generator.generate_response(
            "Complex query needing multiple searches",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify 3 API calls (2 rounds + final without tools)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        # Verify 2 tool executions
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify final call was made without tools
        final_call_args = self.mock_client.messages.create.call_args_list[2]
        self.assertNotIn("tools", final_call_args[1])

        self.assertIn("available information", result)

    def test_empty_tool_results_handling(self):
        """Test handling when tool execution returns empty/null results"""
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "nonexistent topic"}
        mock_tool_content.id = "tool_use_1"

        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_content]
        mock_response_1.stop_reason = "tool_use"

        mock_response_2 = Mock()
        mock_response_2.content = [
            Mock(text="I couldn't find any information about that topic in the course materials.")
        ]
        mock_response_2.stop_reason = "end_turn"

        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        # Mock tool manager returning empty string
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = ""

        result = self.ai_generator.generate_response(
            "Search for nonexistent topic",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify tool was executed and empty result was passed to Claude
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)

        # Verify empty result was included in conversation
        second_call_args = self.mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_result_content = messages[2]["content"][0]["content"]
        self.assertEqual(tool_result_content, "")

        self.assertIn("couldn't find", result)


if __name__ == "__main__":
    unittest.main()
