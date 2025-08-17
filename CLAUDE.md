# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) chatbot system that enables intelligent querying of course materials. The system uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for user interaction.

## Environment Setup

This project requires Python 3.13+ and uses `uv` for dependency management. 

### Prerequisites
- uv package manager
- Anthropic API key (required for Claude AI)

### Installation
```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY
```

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Testing and Development
```bash
# Install new dependencies
uv add <package_name>

# Run with specific Python version
uv run python --version

# Access interactive Python shell with project dependencies
uv run python
```

### Code Quality Tools
```bash
# Format code automatically (recommended before commits)
./scripts/format-code.sh

# Run basic linting checks (fast, good for development)
./scripts/lint-check.sh

# Run comprehensive quality checks (includes type checking)
./scripts/quality-check.sh

# Individual tool commands:
uv run black .                    # Auto-format code
uv run black --check .            # Check formatting without changes
uv run isort .                    # Sort imports
uv run flake8 .                   # Run linter
uv run mypy backend/              # Run type checker
```

### Database Management
The ChromaDB database is stored in `backend/chroma_db/` and is automatically initialized on startup. To reset the database, delete the `chroma_db` directory.

## Architecture Overview

This is a full-stack RAG application with a modular Python backend and vanilla JavaScript frontend.

### Backend Architecture (`/backend/`)

The system follows a layered architecture with clear separation of concerns:

**API Layer (`app.py`)**
- FastAPI application with CORS middleware
- Serves both API endpoints and static frontend files
- Main endpoints: `/api/query` (chat), `/api/courses` (analytics)

**RAG Orchestration (`rag_system.py`)**
- Main coordinator that orchestrates all RAG components
- Manages the interaction between document processing, vector search, and AI generation
- Handles session management and conversation context

**AI Generation (`ai_generator.py`)**
- Interfaces with Anthropic's Claude API
- Uses a tool-based approach where Claude autonomously decides when to search
- Implements conversation history and context management

**Document Processing Pipeline**
- `document_processor.py`: Processes course documents with structured format parsing
- `models.py`: Pydantic models for Course, Lesson, and CourseChunk
- `vector_store.py`: ChromaDB integration with semantic search capabilities

**Search System (`search_tools.py`)**
- Tool-based architecture where AI can call search functions
- `CourseSearchTool`: Semantic search with course/lesson filtering
- `ToolManager`: Manages available tools and tracks search sources

**Session Management (`session_manager.py`)**
- Maintains conversation history per session
- Configurable history length (MAX_HISTORY setting)

### Frontend (`/frontend/`)
- Vanilla HTML/CSS/JavaScript chat interface
- Real-time communication with backend API
- Session persistence and source attribution display

### Configuration (`config.py`)
Centralized configuration using dataclasses:
- API settings (Anthropic model: claude-sonnet-4-20250514)
- Embedding model (all-MiniLM-L6-v2)
- Text processing settings (chunk size: 800, overlap: 100)
- Database paths and limits

## Document Format

The system expects course documents in `/docs/` with this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [lesson_url]
[lesson content...]
```

Documents are automatically processed on startup and stored in ChromaDB with hierarchical metadata (course → lesson → chunks).

## Key Implementation Details

**Tool-Based RAG**: Claude autonomously decides when to search based on query analysis, rather than always searching. This reduces unnecessary API calls and improves response quality.

**Context Preservation**: Text chunks include contextual prefixes indicating their source course and lesson, enabling accurate attribution.

**Session-Based Conversations**: Each chat session maintains conversation history up to MAX_HISTORY messages for context continuity.

**Smart Chunking**: Sentence-based text splitting with configurable overlap preserves semantic meaning across chunk boundaries.

**Error Handling**: Graceful fallbacks throughout the pipeline with informative error messages returned to frontend.

## Access Points

- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Direct API: http://localhost:8000/api/query (POST)

## Recent Updates

### '+ NEW CHAT' Button Implementation
Added a new chat button to the left sidebar that allows users to start fresh conversations without page reload:

**Features:**
- Located above the courses section in the sidebar
- Matches existing sidebar styling (uppercase text, consistent colors)
- Clears current conversation and session state
- Focuses chat input for immediate use
- No page reload required

**Implementation Files:**
- `frontend/index.html`: Added button HTML structure
- `frontend/style.css`: Added `.new-chat-button` styling with hover effects
- `frontend/script.js`: Added `startNewChat()` function and event handler

**Usage:** Click the '+ NEW CHAT' button to clear the current conversation and start a new session while maintaining the same user experience.

## Code Quality Standards

This project maintains high code quality through automated tools and standards:

### Formatting Standards
- **Black**: Code formatter with 100-character line length
- **isort**: Import sorting with black-compatible settings
- All Python code is automatically formatted for consistency

### Linting Standards
- **flake8**: Style guide enforcement with custom configuration
- **mypy**: Type checking for improved code reliability
- Configuration files: `.flake8`, `pyproject.toml`

### Development Workflow
1. **Before committing**: Run `./scripts/format-code.sh` to auto-format code
2. **During development**: Use `./scripts/lint-check.sh` for quick quality checks
3. **Before PR/deployment**: Run `./scripts/quality-check.sh` for comprehensive validation

### Quality Tools Configuration
- **Line length**: 100 characters (more readable than 88, less than 120)
- **Type checking**: Gradual typing with lenient settings for existing code
- **Import organization**: Black-compatible import sorting
- **Error tolerance**: Balanced approach between strict standards and development velocity

The quality tools are configured to work together harmoniously and provide immediate feedback during development.