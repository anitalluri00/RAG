"""
RAG Utilities Package

This package contains utility functions for:
- File processing (PDF, CSV, TXT, XLSX)
- Vector store operations
- Gemini AI integration
"""

__all__ = ["file_processing", "vectorstore"]

# Utility functions
def get_supported_file_types():
    """Return supported file extensions"""
    return {
        'pdf': 'Portable Document Format',
        'csv': 'Comma Separated Values',
        'txt': 'Plain Text',
        'xlsx': 'Excel Spreadsheet',
        'xls': 'Excel 97-2003'
    }

def clean_text(text: str) -> str:
    """Basic text cleaning utility"""
    import re
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

# Version information
__version__ = "1.0.0"