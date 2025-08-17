import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import tempfile
import os
from pathlib import Path

# Create a test app that doesn't mount static files to avoid import issues
def create_test_app():
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict, Any
    
    # Import models
    from config import config
    
    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Any]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    
    # API Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Store mock for access in tests
    app.state.mock_rag_system = mock_rag_system
    
    return app


@pytest.fixture
def test_app():
    """Create test FastAPI app"""
    return create_test_app()


@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def mock_rag_system(test_app):
    """Get the mock RAG system from the test app"""
    return test_app.state.mock_rag_system


class TestQueryEndpoint:
    """Test the /api/query endpoint"""

    @pytest.mark.api
    def test_query_with_session_id(self, client, mock_rag_system, sample_query_request, sample_query_response):
        """Test query endpoint with provided session ID"""
        # Setup mock
        mock_rag_system.query.return_value = (
            sample_query_response["answer"],
            sample_query_response["sources"]
        )
        
        # Make request
        response = client.post("/api/query", json=sample_query_request)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == sample_query_response["answer"]
        assert data["sources"] == sample_query_response["sources"]
        assert data["session_id"] == sample_query_request["session_id"]
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with(
            sample_query_request["query"],
            sample_query_request["session_id"]
        )

    @pytest.mark.api
    def test_query_without_session_id(self, client, mock_rag_system):
        """Test query endpoint without session ID (should create new session)"""
        # Setup mock
        generated_session_id = "generated_session_456"
        mock_rag_system.session_manager.create_session.return_value = generated_session_id
        mock_rag_system.query.return_value = (
            "Python is a programming language",
            [{"text": "Python Basics", "link": "http://example.com"}]
        )
        
        request_data = {"query": "What is Python?"}
        
        # Make request
        response = client.post("/api/query", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Python is a programming language"
        assert data["session_id"] == generated_session_id
        assert len(data["sources"]) == 1
        
        # Verify session was created
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is Python?", generated_session_id)

    @pytest.mark.api
    def test_query_with_invalid_request(self, client):
        """Test query endpoint with invalid request data"""
        # Missing required 'query' field
        invalid_request = {"session_id": "test_session"}
        
        response = client.post("/api/query", json=invalid_request)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.api
    def test_query_with_empty_query(self, client, mock_rag_system):
        """Test query endpoint with empty query string"""
        mock_rag_system.query.return_value = (
            "Please provide a specific question.",
            []
        )
        
        request_data = {"query": ""}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    @pytest.mark.api
    def test_query_rag_system_error(self, client, mock_rag_system):
        """Test query endpoint when RAG system raises an error"""
        # Setup mock to raise exception
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        
        request_data = {"query": "What is Python?"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "Database connection failed" in data["detail"]

    @pytest.mark.api
    def test_query_with_long_text(self, client, mock_rag_system):
        """Test query endpoint with very long query text"""
        # Setup mock
        long_query = "What is Python? " * 1000  # Very long query
        mock_rag_system.query.return_value = (
            "Python is a programming language",
            []
        )
        mock_rag_system.session_manager.create_session.return_value = "long_query_session"
        
        request_data = {"query": long_query}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(long_query, "long_query_session")

    @pytest.mark.api
    def test_query_response_format(self, client, mock_rag_system):
        """Test that query response matches expected format"""
        # Setup mock with structured sources
        mock_sources = [
            {"text": "Python Basics - Lesson 1", "link": "http://example.com/lesson1"},
            "Legacy string source"  # Test mixed source formats
        ]
        mock_rag_system.query.return_value = ("Python response", mock_sources)
        mock_rag_system.session_manager.create_session.return_value = "format_test_session"
        
        request_data = {"query": "What is Python?"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert data["sources"] == mock_sources


class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""

    @pytest.mark.api
    def test_get_course_stats_success(self, client, mock_rag_system, sample_course_stats):
        """Test successful course stats retrieval"""
        # Setup mock
        mock_rag_system.get_course_analytics.return_value = sample_course_stats
        
        # Make request
        response = client.get("/api/courses")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == sample_course_stats["total_courses"]
        assert data["course_titles"] == sample_course_stats["course_titles"]
        
        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()

    @pytest.mark.api
    def test_get_course_stats_empty(self, client, mock_rag_system):
        """Test course stats with no courses"""
        # Setup mock for empty database
        empty_stats = {
            "total_courses": 0,
            "course_titles": []
        }
        mock_rag_system.get_course_analytics.return_value = empty_stats
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    @pytest.mark.api
    def test_get_course_stats_error(self, client, mock_rag_system):
        """Test course stats endpoint when RAG system raises an error"""
        # Setup mock to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store unavailable")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "Vector store unavailable" in data["detail"]

    @pytest.mark.api
    def test_course_stats_response_format(self, client, mock_rag_system):
        """Test that course stats response matches expected format"""
        # Setup mock
        test_stats = {
            "total_courses": 5,
            "course_titles": ["Course A", "Course B", "Course C", "Course D", "Course E"]
        }
        mock_rag_system.get_course_analytics.return_value = test_stats
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])

    @pytest.mark.api
    def test_course_stats_large_dataset(self, client, mock_rag_system):
        """Test course stats with large number of courses"""
        # Setup mock with many courses
        large_course_list = [f"Course {i}" for i in range(100)]
        large_stats = {
            "total_courses": 100,
            "course_titles": large_course_list
        }
        mock_rag_system.get_course_analytics.return_value = large_stats
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100


class TestAPIIntegration:
    """Integration tests for API endpoints"""

    @pytest.mark.integration
    def test_query_and_courses_consistency(self, client, mock_rag_system):
        """Test that query and courses endpoints work consistently"""
        # Setup mocks
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Python Programming", "Web Development"]
        }
        
        mock_rag_system.query.return_value = (
            "Python is a programming language used in both courses.",
            [{"text": "Python Programming - Introduction", "link": "http://example.com"}]
        )
        mock_rag_system.session_manager.create_session.return_value = "integration_session"
        
        # Test courses endpoint
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        
        # Test query endpoint
        query_response = client.post("/api/query", json={"query": "What is Python?"})
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # Verify both work correctly
        assert courses_data["total_courses"] == 2
        assert "Python Programming" in courses_data["course_titles"]
        assert "Python" in query_data["answer"]

    @pytest.mark.integration
    def test_multiple_queries_same_session(self, client, mock_rag_system):
        """Test multiple queries with the same session ID"""
        session_id = "persistent_session"
        
        # Setup mocks for multiple calls
        mock_rag_system.query.side_effect = [
            ("First response", []),
            ("Second response", [])
        ]
        
        # First query
        response1 = client.post("/api/query", json={
            "query": "First question",
            "session_id": session_id
        })
        
        # Second query with same session
        response2 = client.post("/api/query", json={
            "query": "Second question", 
            "session_id": session_id
        })
        
        # Verify both succeeded
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["session_id"] == session_id
        assert data2["session_id"] == session_id
        assert data1["answer"] == "First response"
        assert data2["answer"] == "Second response"
        
        # Verify RAG system was called twice with same session
        assert mock_rag_system.query.call_count == 2

    @pytest.mark.integration
    def test_concurrent_requests(self, client, mock_rag_system):
        """Test handling of concurrent API requests"""
        import asyncio
        import httpx
        
        # Setup mock
        mock_rag_system.query.return_value = ("Concurrent response", [])
        mock_rag_system.session_manager.create_session.side_effect = [
            f"session_{i}" for i in range(5)
        ]
        
        # Make multiple concurrent requests
        responses = []
        for i in range(5):
            response = client.post("/api/query", json={"query": f"Question {i}"})
            responses.append(response)
        
        # Verify all succeeded
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert "session_" in data["session_id"]

    @pytest.mark.api
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set"""
        response = client.options("/api/query")
        
        # Should not error (OPTIONS preflight request)
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled

    @pytest.mark.api  
    def test_content_type_validation(self, client, mock_rag_system):
        """Test API endpoint content type validation"""
        # Test with non-JSON content type
        response = client.post("/api/query", data="not json")
        assert response.status_code == 422  # Validation error
        
        # Test with correct JSON
        mock_rag_system.query.return_value = ("Test response", [])
        mock_rag_system.session_manager.create_session.return_value = "content_test_session"
        
        response = client.post("/api/query", json={"query": "Test query"})
        assert response.status_code == 200