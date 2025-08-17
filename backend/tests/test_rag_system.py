import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import Config


class TestRAGSystemContentQueries(unittest.TestCase):
    """Test cases for RAG system content query handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a test config
        self.config = Config()
        self.config.ANTHROPIC_API_KEY = "test_key"
        self.config.ANTHROPIC_MODEL = "claude-3-haiku-20240307"
        self.config.CHROMA_PATH = "./test_chroma_db"
        self.config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.config.MAX_RESULTS = 5  # Set to non-zero for testing
        self.config.CHUNK_SIZE = 800
        self.config.CHUNK_OVERLAP = 100
        self.config.MAX_HISTORY = 2
        
        # Mock all the dependencies
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr:
            
            self.mock_doc_processor = Mock()
            self.mock_vector_store = Mock()
            self.mock_ai_generator = Mock()
            self.mock_session_manager = Mock()
            
            mock_doc_proc.return_value = self.mock_doc_processor
            mock_vector_store.return_value = self.mock_vector_store
            mock_ai_gen.return_value = self.mock_ai_generator
            mock_session_mgr.return_value = self.mock_session_manager
            
            self.rag_system = RAGSystem(self.config)
    
    def test_content_query_successful_flow(self):
        """Test successful content query with tool execution"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "Python is a high-level programming language known for its simplicity and readability."
        
        # Mock tool manager sources
        mock_sources = [
            {"text": "Python Basics - Lesson 1", "link": "http://example.com/lesson1"},
            {"text": "Python Basics - Lesson 2", "link": "http://example.com/lesson2"}
        ]
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        response, sources = self.rag_system.query("What is Python?")
        
        # Verify AI generator was called with correct parameters
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args
        
        # Check the query parameter
        self.assertIn("What is Python?", call_args[1]["query"])
        
        # Check that tools were provided
        self.assertIsNotNone(call_args[1]["tools"])
        self.assertIsNotNone(call_args[1]["tool_manager"])
        
        # Verify response and sources
        self.assertEqual(response, "Python is a high-level programming language known for its simplicity and readability.")
        self.assertEqual(sources, mock_sources)
        
        # Verify sources were reset after retrieval
        self.rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_content_query_with_session_history(self):
        """Test content query with conversation history"""
        # Mock session history
        self.mock_session_manager.get_conversation_history.return_value = "User: Hello\nAssistant: Hi! How can I help you with course materials?"
        
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "Based on our previous conversation, here's more about Python..."
        
        # Mock empty sources
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query with session
        response, sources = self.rag_system.query("Tell me more about Python", session_id="session_123")
        
        # Verify session history was retrieved
        self.mock_session_manager.get_conversation_history.assert_called_once_with("session_123")
        
        # Verify AI generator received history
        call_args = self.mock_ai_generator.generate_response.call_args
        self.assertEqual(call_args[1]["conversation_history"], "User: Hello\nAssistant: Hi! How can I help you with course materials?")
        
        # Verify session was updated
        self.mock_session_manager.add_exchange.assert_called_once_with(
            "session_123",
            "Answer this question about course materials: Tell me more about Python",
            "Based on our previous conversation, here's more about Python..."
        )
    
    def test_content_query_no_session(self):
        """Test content query without session ID"""
        self.mock_ai_generator.generate_response.return_value = "Python response"
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = self.rag_system.query("What is Python?")
        
        # Verify no session operations were called
        self.mock_session_manager.get_conversation_history.assert_not_called()
        self.mock_session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator was called without history
        call_args = self.mock_ai_generator.generate_response.call_args
        self.assertIsNone(call_args[1]["conversation_history"])
    
    def test_query_prompt_formatting(self):
        """Test that query is properly formatted for AI"""
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        self.rag_system.query("How do Python functions work?")
        
        call_args = self.mock_ai_generator.generate_response.call_args
        query_param = call_args[1]["query"]
        
        # Verify query is formatted with instruction
        self.assertIn("Answer this question about course materials:", query_param)
        self.assertIn("How do Python functions work?", query_param)
    
    def test_tool_manager_initialization(self):
        """Test that tool manager is properly initialized with tools"""
        # Verify tool manager has the expected tools registered
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        
        # Should have CourseSearchTool and CourseOutlineTool
        tool_names = [tool["name"] for tool in tool_definitions]
        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)
    
    def test_analytics_integration(self):
        """Test course analytics functionality"""
        # Mock vector store responses
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Python Basics", "Advanced Python", "Data Science", "Machine Learning", "Web Development"
        ]
        
        analytics = self.rag_system.get_course_analytics()
        
        # Verify analytics structure
        self.assertEqual(analytics["total_courses"], 5)
        self.assertEqual(len(analytics["course_titles"]), 5)
        self.assertIn("Python Basics", analytics["course_titles"])


class TestRAGSystemComponentIntegration(unittest.TestCase):
    """Test integration between RAG system components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.config.ANTHROPIC_API_KEY = "test_key"
        self.config.MAX_RESULTS = 3  # Set to non-zero
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            self.rag_system = RAGSystem(self.config)
    
    def test_component_initialization_with_config(self):
        """Test that all components are initialized with correct config values"""
        # Verify RAG system stores config
        self.assertEqual(self.rag_system.config, self.config)
        
        # Verify components exist
        self.assertIsNotNone(self.rag_system.document_processor)
        self.assertIsNotNone(self.rag_system.vector_store)
        self.assertIsNotNone(self.rag_system.ai_generator)
        self.assertIsNotNone(self.rag_system.session_manager)
        self.assertIsNotNone(self.rag_system.tool_manager)
        self.assertIsNotNone(self.rag_system.search_tool)
        self.assertIsNotNone(self.rag_system.outline_tool)
    
    def test_search_tool_vector_store_integration(self):
        """Test that search tools are properly connected to vector store"""
        # Verify search tool has access to vector store
        self.assertEqual(self.rag_system.search_tool.store, self.rag_system.vector_store)
        self.assertEqual(self.rag_system.outline_tool.store, self.rag_system.vector_store)
    
    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.listdir')
    def test_document_processing_integration(self, mock_listdir, mock_exists):
        """Test document processing and vector store integration"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf"]
        
        # Mock document processor
        from models import Course, Lesson, CourseChunk
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://test.com",
            lessons=[]
        )
        mock_chunks = [
            CourseChunk(
                content="Test content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        self.rag_system.document_processor.process_course_document.return_value = (mock_course, mock_chunks)
        self.rag_system.vector_store.get_existing_course_titles.return_value = []
        
        # Test adding course folder
        total_courses, total_chunks = self.rag_system.add_course_folder("./test_docs")
        
        # Verify integration
        self.assertEqual(total_courses, 2)  # Two files processed
        self.assertEqual(total_chunks, 2)   # Two chunks created
        
        # Verify vector store methods were called
        self.assertEqual(self.rag_system.vector_store.add_course_metadata.call_count, 2)
        self.assertEqual(self.rag_system.vector_store.add_course_content.call_count, 2)


class TestRAGSystemErrorHandling(unittest.TestCase):
    """Test error handling in RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.config.ANTHROPIC_API_KEY = "test_key"
        self.config.MAX_RESULTS = 0  # Test the zero results issue
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            self.rag_system = RAGSystem(self.config)
    
    def test_zero_max_results_configuration(self):
        """Test that zero MAX_RESULTS causes search issues"""
        # This test demonstrates the configuration issue
        # The vector store should be initialized with MAX_RESULTS = 0
        # which will cause search to return no results
        
        # Mock vector store to return empty results due to zero limit
        from vector_store import SearchResults
        empty_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        self.rag_system.vector_store.search.return_value = empty_results
        
        # Execute search tool directly
        result = self.rag_system.search_tool.execute("test query")
        
        # Should return "no content found" message
        self.assertEqual(result, "No relevant content found.")
        
        # Verify search was called with no limit (which defaults to 0)
        self.rag_system.vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
    
    def test_ai_generator_error_handling(self):
        """Test handling of AI generator errors"""
        # Mock AI generator to raise exception
        self.rag_system.ai_generator.generate_response.side_effect = Exception("API Error")
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Query should handle the exception gracefully
        with self.assertRaises(Exception):
            self.rag_system.query("What is Python?")
    
    def test_document_processing_error_handling(self):
        """Test handling of document processing errors"""
        # Mock document processor to raise exception
        self.rag_system.document_processor.process_course_document.side_effect = Exception("File read error")
        
        # Add document should handle error gracefully
        course, chunks = self.rag_system.add_course_document("nonexistent_file.txt")
        
        # Should return None and 0 for error case
        self.assertIsNone(course)
        self.assertEqual(chunks, 0)


class TestRAGSystemWithRealisticConfig(unittest.TestCase):
    """Test RAG system with realistic configuration that should work"""
    
    def setUp(self):
        """Set up test fixtures with working configuration"""
        self.config = Config()
        self.config.ANTHROPIC_API_KEY = "test_key"
        self.config.MAX_RESULTS = 5  # Fix the zero results issue
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            self.rag_system = RAGSystem(self.config)
    
    def test_working_content_query_flow(self):
        """Test that content queries work with proper configuration"""
        # Mock vector store to return actual results
        from vector_store import SearchResults
        results = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.rag_system.vector_store.search.return_value = results
        self.rag_system.vector_store.get_lesson_link.return_value = "http://example.com"
        
        # Mock AI generator to simulate tool use
        self.rag_system.ai_generator.generate_response.return_value = "Python is a high-level programming language"
        
        # Mock tool manager
        sources = [{"text": "Python Basics - Lesson 1", "link": "http://example.com"}]
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=sources)
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        response, returned_sources = self.rag_system.query("What is Python?")
        
        # Verify successful flow
        self.assertEqual(response, "Python is a high-level programming language")
        self.assertEqual(returned_sources, sources)
        
        # Verify components were called
        self.rag_system.ai_generator.generate_response.assert_called_once()
        self.rag_system.tool_manager.get_last_sources.assert_called_once()
        self.rag_system.tool_manager.reset_sources.assert_called_once()


if __name__ == "__main__":
    unittest.main()