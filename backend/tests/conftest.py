import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import Config
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


@pytest.fixture
def test_config():
    """Create a test configuration with safe values"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test_key"
    config.ANTHROPIC_MODEL = "claude-3-haiku-20240307"
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_course():
    """Create a mock course object"""
    lessons = [
        Lesson(
            lesson_number=0,
            title="Introduction to Python",
            content="Python is a high-level programming language.",
            lesson_link="http://example.com/lesson0"
        ),
        Lesson(
            lesson_number=1,
            title="Python Basics", 
            content="Variables and data types in Python.",
            lesson_link="http://example.com/lesson1"
        )
    ]
    
    return Course(
        title="Python Programming",
        instructor="John Doe",
        course_link="http://example.com/course",
        lessons=lessons
    )


@pytest.fixture
def mock_course_chunks():
    """Create mock course chunks"""
    return [
        CourseChunk(
            content="Python is a high-level programming language known for its simplicity.",
            course_title="Python Programming",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables in Python can store different types of data.",
            course_title="Python Programming", 
            lesson_number=1,
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_search_results():
    """Create mock search results from vector store"""
    from vector_store import SearchResults
    return SearchResults(
        documents=[
            "Python is a high-level programming language",
            "Variables in Python store data"
        ],
        metadata=[
            {"course_title": "Python Programming", "lesson_number": 0},
            {"course_title": "Python Programming", "lesson_number": 1}
        ],
        distances=[0.1, 0.2],
        error=None
    )


@pytest.fixture
def mock_rag_system(test_config):
    """Create a RAG system with all dependencies mocked"""
    with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
         patch('rag_system.VectorStore') as mock_vector_store, \
         patch('rag_system.AIGenerator') as mock_ai_gen, \
         patch('rag_system.SessionManager') as mock_session_mgr:
        
        # Create mock instances
        mock_doc_processor = Mock()
        mock_vector_store_instance = Mock()
        mock_ai_generator = Mock()
        mock_session_manager = Mock()
        
        # Configure mock constructors
        mock_doc_proc.return_value = mock_doc_processor
        mock_vector_store.return_value = mock_vector_store_instance
        mock_ai_gen.return_value = mock_ai_generator
        mock_session_mgr.return_value = mock_session_manager
        
        # Initialize RAG system
        rag_system = RAGSystem(test_config)
        
        # Attach mocks for easy access in tests
        rag_system.mock_doc_processor = mock_doc_processor
        rag_system.mock_vector_store = mock_vector_store_instance
        rag_system.mock_ai_generator = mock_ai_generator
        rag_system.mock_session_manager = mock_session_manager
        
        yield rag_system


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is Python?",
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response data"""
    return {
        "answer": "Python is a high-level programming language known for its simplicity and readability.",
        "sources": [
            {"text": "Python Programming - Lesson 0", "link": "http://example.com/lesson0"},
            {"text": "Python Programming - Lesson 1", "link": "http://example.com/lesson1"}
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_course_stats():
    """Sample course statistics"""
    return {
        "total_courses": 3,
        "course_titles": ["Python Programming", "Web Development", "Data Science"]
    }


@pytest.fixture 
def mock_env_vars():
    """Mock environment variables for testing"""
    env_vars = {
        "ANTHROPIC_API_KEY": "test_api_key"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def course_document_content():
    """Sample course document content for testing"""
    return """Course Title: Python Programming Fundamentals
Course Link: https://example.com/python-course
Course Instructor: Dr. Jane Smith

Lesson 0: Introduction to Programming
Lesson Link: https://example.com/python-course/lesson-0

Welcome to Python programming! Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

Key features of Python:
- Easy to learn and use
- Interpreted language
- Cross-platform compatibility
- Large standard library
- Strong community support

Lesson 1: Variables and Data Types  
Lesson Link: https://example.com/python-course/lesson-1

In Python, variables are used to store data values. Unlike other programming languages, Python has no command for declaring a variable. A variable is created the moment you first assign a value to it.

Common data types in Python:
- Integer (int): Whole numbers
- Float: Decimal numbers  
- String (str): Text data
- Boolean (bool): True/False values
- List: Ordered collection of items
"""


@pytest.fixture
def frontend_files(temp_directory):
    """Create mock frontend files for testing static file serving"""
    frontend_dir = Path(temp_directory) / "frontend"
    frontend_dir.mkdir()
    
    # Create index.html
    index_file = frontend_dir / "index.html"
    index_file.write_text("""
    <!DOCTYPE html>
    <html>
    <head><title>RAG Chatbot</title></head>
    <body>
        <h1>RAG Chatbot Interface</h1>
        <div id="chat-container"></div>
    </body>
    </html>
    """)
    
    # Create style.css
    style_file = frontend_dir / "style.css"
    style_file.write_text("""
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 20px;
    }
    #chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    """)
    
    return frontend_dir


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress chromadb and other warnings during testing"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)