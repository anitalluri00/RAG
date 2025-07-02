"""
RAG Application Package

This package contains the main application components for the RAG system with Gemini AI.
"""

__version__ = "1.0.0"
__all__ = ["main", "utils"]

# Initialize package-level variables
app_name = "RAG-Gemini"
app_description = "Retrieval-Augmented Generation System using Google Gemini AI"

# Package initialization code
def initialize():
    """Initialize application components"""
    # Create necessary directories
    import os
    os.makedirs("uploads", exist_ok=True)
    
    # Verify environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Run initialization when package is imported
initialize()