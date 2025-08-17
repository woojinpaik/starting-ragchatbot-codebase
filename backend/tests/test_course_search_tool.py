import os
import sys
import unittest
from unittest.mock import MagicMock, Mock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool.execute method"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_successful_search(self):
        """Test successful search with results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Course content about Python", "More Python content"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        # Execute search
        result = self.search_tool.execute("python basics")

        # Verify search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="python basics", course_name=None, lesson_number=None
        )

        # Verify result contains formatted content
        self.assertIn("[Python Basics - Lesson 1]", result)
        self.assertIn("Course content about Python", result)
        self.assertIn("[Python Basics - Lesson 2]", result)
        self.assertIn("More Python content", result)

        # Verify sources were tracked
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertEqual(self.search_tool.last_sources[0]["text"], "Python Basics - Lesson 1")
        self.assertEqual(self.search_tool.last_sources[0]["link"], "http://example.com/lesson1")

    def test_execute_with_course_filter(self):
        """Test search with course name filter"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        result = self.search_tool.execute("loops", course_name="Advanced Python")

        # Verify search was called with course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="loops", course_name="Advanced Python", lesson_number=None
        )

        self.assertIn("[Advanced Python - Lesson 3]", result)
        self.assertIn("Filtered content", result)

    def test_execute_with_lesson_filter(self):
        """Test search with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 5}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson5"

        result = self.search_tool.execute("variables", lesson_number=5)

        # Verify search was called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="variables", course_name=None, lesson_number=5
        )

        self.assertIn("[Python Basics - Lesson 5]", result)
        self.assertIn("Lesson specific content", result)

    def test_execute_with_both_filters(self):
        """Test search with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Highly specific content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 2}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "http://example.com/ds/lesson2"

        result = self.search_tool.execute("pandas", course_name="Data Science", lesson_number=2)

        # Verify search was called with both filters
        self.mock_vector_store.search.assert_called_once_with(
            query="pandas", course_name="Data Science", lesson_number=2
        )

        self.assertIn("[Data Science - Lesson 2]", result)
        self.assertIn("Highly specific content", result)

    def test_execute_search_error(self):
        """Test handling of search errors"""
        # Mock search error
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("anything")

        # Verify error is returned
        self.assertEqual(result, "Database connection failed")

        # Verify no sources are set for error case
        self.assertEqual(len(self.search_tool.last_sources), 0)

    def test_execute_empty_results(self):
        """Test handling of empty search results"""
        # Mock empty results
        mock_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("nonexistent topic")

        # Verify appropriate empty message
        self.assertEqual(result, "No relevant content found.")

    def test_execute_empty_results_with_course_filter(self):
        """Test empty results message includes filter info"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("topic", course_name="Nonexistent Course")

        self.assertEqual(result, "No relevant content found in course 'Nonexistent Course'.")

    def test_execute_empty_results_with_lesson_filter(self):
        """Test empty results message includes lesson filter info"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("topic", lesson_number=99)

        self.assertEqual(result, "No relevant content found in lesson 99.")

    def test_execute_empty_results_with_both_filters(self):
        """Test empty results message includes both filter info"""
        mock_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("topic", course_name="Test Course", lesson_number=5)

        self.assertEqual(result, "No relevant content found in course 'Test Course' in lesson 5.")

    def test_execute_metadata_handling_edge_cases(self):
        """Test handling of missing metadata fields"""
        # Test with missing course_title
        mock_results = SearchResults(
            documents=["Content without proper metadata"],
            metadata=[{"lesson_number": 1}],  # Missing course_title
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        result = self.search_tool.execute("test")

        # Should handle missing course_title gracefully
        self.assertIn("[unknown - Lesson 1]", result)
        self.assertIn("Content without proper metadata", result)

    def test_execute_metadata_missing_lesson_number(self):
        """Test handling of missing lesson number in metadata"""
        mock_results = SearchResults(
            documents=["Content without lesson number"],
            metadata=[{"course_title": "Test Course"}],  # Missing lesson_number
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        result = self.search_tool.execute("test")

        # Should handle missing lesson_number gracefully
        self.assertIn("[Test Course]", result)
        self.assertNotIn("Lesson", result.split("[Test Course]")[1].split("]")[0])
        self.assertIn("Content without lesson number", result)


class TestCourseSearchToolIntegration(unittest.TestCase):
    """Integration tests that test the tool definition and parameter passing"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted"""
        definition = self.search_tool.get_tool_definition()

        # Verify structure
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)

        # Verify schema structure
        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertEqual(schema["required"], ["query"])

        # Verify properties
        properties = schema["properties"]
        self.assertIn("query", properties)
        self.assertIn("course_name", properties)
        self.assertIn("lesson_number", properties)

        # Verify query is required and others are optional
        self.assertEqual(properties["query"]["type"], "string")
        self.assertEqual(properties["course_name"]["type"], "string")
        self.assertEqual(properties["lesson_number"]["type"], "integer")


if __name__ == "__main__":
    unittest.main()
