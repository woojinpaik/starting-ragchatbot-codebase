#!/usr/bin/env python3
"""
Verification test to confirm the MAX_RESULTS fix resolves the 'query failed' issue.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool


class TestFixVerification(unittest.TestCase):
    """Verify that the MAX_RESULTS fix resolves the query failed issue"""
    
    def test_config_fix_applied(self):
        """Test that MAX_RESULTS is no longer zero"""
        config = Config()
        self.assertGreater(config.MAX_RESULTS, 0, 
                          "MAX_RESULTS should be greater than 0 after fix")
        self.assertEqual(config.MAX_RESULTS, 5, 
                        "MAX_RESULTS should be set to 5")
    
    def test_vector_store_with_fixed_config(self):
        """Test that VectorStore now uses non-zero max_results"""
        config = Config()
        
        with patch('vector_store.chromadb.PersistentClient') as mock_client, \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Create vector store with fixed config
            vector_store = VectorStore(
                chroma_path="./test_db",
                embedding_model="test-model",
                max_results=config.MAX_RESULTS
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
            
            # Verify that query was called with n_results=5 (not 0)
            mock_collection.query.assert_called_once_with(
                query_texts=['test query'],
                n_results=5,  # This should now be 5, not 0
                where=None
            )
            
            # Verify results are returned
            self.assertEqual(len(results.documents), 2)
            self.assertIsNone(results.error)
    
    def test_search_tool_with_fixed_config(self):
        """Test that CourseSearchTool now returns content instead of 'No content found'"""
        # Mock vector store with successful results
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["Python is a programming language", "Variables store data"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
        
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Execute search
        result = search_tool.execute("What is Python?")
        
        # Should now return actual content, not "No relevant content found"
        self.assertNotEqual(result, "No relevant content found.")
        self.assertIn("Python is a programming language", result)
        self.assertIn("[Python Basics - Lesson 1]", result)
        self.assertIn("[Python Basics - Lesson 2]", result)
        
        # Verify sources were tracked
        self.assertEqual(len(search_tool.last_sources), 2)
    
    def test_end_to_end_fix_simulation(self):
        """Simulate the full RAG system flow with the fix"""
        from rag_system import RAGSystem
        from ai_generator import AIGenerator
        
        # Create config with the fix
        config = Config()
        self.assertEqual(config.MAX_RESULTS, 5)  # Confirm fix is applied
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_gen_class, \
             patch('rag_system.SessionManager'):
            
            # Mock vector store to return content (simulating successful search)
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = SearchResults(
                documents=["Python is a high-level programming language"],
                metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
                distances=[0.1],
                error=None
            )
            mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
            mock_vector_store_class.return_value = mock_vector_store
            
            # Mock AI generator
            mock_ai_generator = Mock()
            mock_ai_generator.generate_response.return_value = "Python is a high-level programming language used for web development, data science, and more."
            mock_ai_gen_class.return_value = mock_ai_generator
            
            # Create RAG system
            rag_system = RAGSystem(config)
            
            # Verify vector store was initialized with MAX_RESULTS=5
            mock_vector_store_class.assert_called_once_with(
                config.CHROMA_PATH,
                config.EMBEDDING_MODEL,
                5  # Should be 5, not 0
            )
            
            # Mock tool manager sources
            sources = [{"text": "Python Basics - Lesson 1", "link": "http://example.com/lesson1"}]
            rag_system.tool_manager.get_last_sources = Mock(return_value=sources)
            rag_system.tool_manager.reset_sources = Mock()
            
            # Execute query
            response, returned_sources = rag_system.query("What is Python?")
            
            # Should now get a real response, not query failed
            self.assertNotEqual(response, "query failed")
            self.assertIn("Python", response)
            self.assertEqual(returned_sources, sources)


class TestRegressionPrevention(unittest.TestCase):
    """Tests to prevent regression of the configuration issue"""
    
    def test_max_results_never_zero(self):
        """Ensure MAX_RESULTS is never accidentally set to 0 again"""
        config = Config()
        self.assertGreater(config.MAX_RESULTS, 0, 
                          "MAX_RESULTS must never be 0 - this causes query failures")
    
    def test_reasonable_max_results_value(self):
        """Ensure MAX_RESULTS is set to a reasonable value"""
        config = Config()
        self.assertGreaterEqual(config.MAX_RESULTS, 3, 
                               "MAX_RESULTS should be at least 3 for meaningful search")
        self.assertLessEqual(config.MAX_RESULTS, 20, 
                           "MAX_RESULTS should not be too high to avoid performance issues")
    
    def test_other_config_values_intact(self):
        """Ensure other configuration values weren't broken by the fix"""
        config = Config()
        
        # Verify other settings are still correct
        self.assertEqual(config.ANTHROPIC_MODEL, "claude-sonnet-4-20250514")
        self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(config.CHUNK_SIZE, 800)
        self.assertEqual(config.CHUNK_OVERLAP, 100)
        self.assertEqual(config.MAX_HISTORY, 2)
        self.assertEqual(config.CHROMA_PATH, "./chroma_db")


if __name__ == "__main__":
    unittest.main()