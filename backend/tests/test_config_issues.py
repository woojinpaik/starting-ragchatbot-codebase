import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from vector_store import VectorStore, SearchResults


class TestConfigurationIssues(unittest.TestCase):
    """Test cases to identify configuration issues causing 'query failed'"""
    
    def test_max_results_zero_issue(self):
        """Test that MAX_RESULTS = 0 causes search to return no results"""
        config = Config()
        
        # Verify the problematic configuration
        self.assertEqual(config.MAX_RESULTS, 0, 
                        "MAX_RESULTS should be 0 to demonstrate the issue")
        
        # Mock ChromaDB components
        with patch('vector_store.chromadb.PersistentClient') as mock_client, \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Create vector store with problematic config
            vector_store = VectorStore(
                chroma_path="./test_db",
                embedding_model="test-model",
                max_results=config.MAX_RESULTS  # This is 0!
            )
            
            # Mock ChromaDB to return some results
            mock_collection.query.return_value = {
                'documents': [['Document 1', 'Document 2']],
                'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1},
                              {'course_title': 'Test Course', 'lesson_number': 2}]],
                'distances': [[0.1, 0.2]]
            }
            
            # Perform search - this should fail because n_results=0
            results = vector_store.search("test query")
            
            # Verify that query was called with n_results=0
            mock_collection.query.assert_called_once_with(
                query_texts=['test query'],
                n_results=0,  # This is the problem!
                where=None
            )
            
            # This should return empty results or cause an error
            # The exact behavior depends on ChromaDB implementation
    
    def test_max_results_fix(self):
        """Test that setting MAX_RESULTS > 0 fixes the issue"""
        # Create config with fixed value
        config = Config()
        config.MAX_RESULTS = 5  # Fix the issue
        
        with patch('vector_store.chromadb.PersistentClient') as mock_client, \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Create vector store with fixed config
            vector_store = VectorStore(
                chroma_path="./test_db",
                embedding_model="test-model",
                max_results=config.MAX_RESULTS  # This is now 5
            )
            
            # Mock ChromaDB to return results
            mock_collection.query.return_value = {
                'documents': [['Document 1', 'Document 2']],
                'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1},
                              {'course_title': 'Test Course', 'lesson_number': 2}]],
                'distances': [[0.1, 0.2]]
            }
            
            # Perform search
            results = vector_store.search("test query")
            
            # Verify that query was called with n_results=5
            mock_collection.query.assert_called_once_with(
                query_texts=['test query'],
                n_results=5,  # This should work!
                where=None
            )
            
            # Verify results are returned
            self.assertEqual(len(results.documents), 2)
            self.assertEqual(results.documents[0], 'Document 1')
            self.assertIsNone(results.error)
    
    def test_search_with_limit_override(self):
        """Test that providing explicit limit overrides MAX_RESULTS"""
        config = Config()
        # Even with MAX_RESULTS = 0, explicit limit should work
        
        with patch('vector_store.chromadb.PersistentClient') as mock_client, \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            vector_store = VectorStore(
                chroma_path="./test_db",
                embedding_model="test-model",
                max_results=config.MAX_RESULTS  # Still 0
            )
            
            mock_collection.query.return_value = {
                'documents': [['Document 1']],
                'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
                'distances': [[0.1]]
            }
            
            # Search with explicit limit
            results = vector_store.search("test query", limit=3)
            
            # Should use the explicit limit, not MAX_RESULTS
            mock_collection.query.assert_called_once_with(
                query_texts=['test query'],
                n_results=3,  # Uses explicit limit
                where=None
            )
    
    def test_anthropic_api_key_configuration(self):
        """Test Anthropic API key configuration"""
        config = Config()
        
        # In real environment, this should be set
        # In test environment, it might be empty
        if config.ANTHROPIC_API_KEY:
            self.assertIsInstance(config.ANTHROPIC_API_KEY, str)
            self.assertGreater(len(config.ANTHROPIC_API_KEY), 0)
        else:
            # If not set, this could cause AI generation to fail
            print("WARNING: ANTHROPIC_API_KEY is not set in config")
    
    def test_model_configuration(self):
        """Test AI model configuration"""
        config = Config()
        
        # Verify model is set to expected value
        self.assertEqual(config.ANTHROPIC_MODEL, "claude-sonnet-4-20250514")
        
        # Verify other settings
        self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(config.CHUNK_SIZE, 800)
        self.assertEqual(config.CHUNK_OVERLAP, 100)
        self.assertEqual(config.MAX_HISTORY, 2)
        self.assertEqual(config.CHROMA_PATH, "./chroma_db")


class TestSystemIntegrationIssues(unittest.TestCase):
    """Test potential integration issues that could cause 'query failed'"""
    
    def test_search_tool_with_zero_results_config(self):
        """Test CourseSearchTool behavior with zero MAX_RESULTS"""
        from search_tools import CourseSearchTool
        
        # Mock vector store that uses zero max_results
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Execute search
        result = search_tool.execute("test query")
        
        # Should return "no content found" message
        self.assertEqual(result, "No relevant content found.")
        
        # This demonstrates the issue: search returns no results
        # because the underlying vector store is configured with max_results=0
    
    def test_ai_generator_with_empty_tool_results(self):
        """Test AI generator behavior when tools return empty results"""
        from ai_generator import AIGenerator
        from search_tools import ToolManager, CourseSearchTool
        
        # Mock Anthropic client
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            # Mock tool use response
            mock_tool_content = Mock()
            mock_tool_content.type = "tool_use"
            mock_tool_content.name = "search_course_content"
            mock_tool_content.input = {"query": "test"}
            mock_tool_content.id = "tool_123"
            
            mock_initial_response = Mock()
            mock_initial_response.content = [mock_tool_content]
            mock_initial_response.stop_reason = "tool_use"
            
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="I couldn't find any relevant information")]
            
            mock_client.messages.create.side_effect = [
                mock_initial_response,
                mock_final_response
            ]
            
            # Create AI generator
            ai_generator = AIGenerator("test_key", "test_model")
            
            # Mock tool manager with search tool that returns empty results
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = SearchResults(
                documents=[], metadata=[], distances=[], error=None
            )
            
            search_tool = CourseSearchTool(mock_vector_store)
            tool_manager = ToolManager()
            tool_manager.register_tool(search_tool)
            
            # Generate response
            result = ai_generator.generate_response(
                "What is Python?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )
            
            # Verify tool was executed and returned empty results
            mock_vector_store.search.assert_called_once()
            
            # AI should handle empty tool results gracefully
            self.assertEqual(result, "I couldn't find any relevant information")
    
    def test_full_system_with_configuration_issue(self):
        """Test full RAG system with the configuration issue"""
        from rag_system import RAGSystem
        
        # Create config with the issue
        config = Config()
        self.assertEqual(config.MAX_RESULTS, 0)  # Confirm the issue exists
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_gen_class, \
             patch('rag_system.SessionManager'):
            
            # Mock vector store to be initialized with MAX_RESULTS=0
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store
            
            mock_ai_generator = Mock()
            mock_ai_gen_class.return_value = mock_ai_generator
            
            # Create RAG system
            rag_system = RAGSystem(config)
            
            # Verify vector store was initialized with problematic config
            mock_vector_store_class.assert_called_once_with(
                config.CHROMA_PATH,
                config.EMBEDDING_MODEL,
                config.MAX_RESULTS  # This is 0!
            )
            
            # This demonstrates the root cause of the issue


if __name__ == "__main__":
    unittest.main()