"""
Utility Functions for Customer Support RAG System
Common helper functions and utilities
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import re
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file safely
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data or empty dict if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {}

def save_json_file(data: Any, file_path: str, create_dirs: bool = True):
    """
    Save data to JSON file safely
    
    Args:
        data: Data to save
        file_path: Output file path
        create_dirs: Whether to create directories if they don't exist
    """
    try:
        if create_dirs:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        raise

def generate_customer_id(email: Optional[str] = None, phone: Optional[str] = None) -> str:
    """
    Generate a consistent customer ID from contact information
    
    Args:
        email: Customer email
        phone: Customer phone number
        
    Returns:
        Generated customer ID
    """
    if email:
        # Use email as primary identifier
        identifier = email.lower().strip()
    elif phone:
        # Clean phone number and use it
        identifier = re.sub(r'[^\d]', '', phone)
    else:
        # Generate random ID based on timestamp
        identifier = f"customer_{int(time.time())}"
    
    # Generate hash for privacy
    return hashlib.md5(identifier.encode()).hexdigest()[:12]

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
    
    # Remove multiple punctuation
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 characters
            search_start = max(end - 100, start)
            sentence_end = text.rfind('.', search_start, end)
            
            if sentence_end > search_start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(end - overlap, start + 1)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using word overlap
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def format_conversation_history(messages: List[Dict[str, Any]]) -> str:
    """
    Format conversation history for display
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Formatted conversation string
    """
    if not messages:
        return "No previous conversation history."
    
    formatted_messages = []
    
    for i, msg in enumerate(messages):
        timestamp = msg.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        speaker = msg.get('speaker', 'Customer')
        content = msg.get('content', msg.get('message', ''))
        
        time_str = timestamp.strftime('%H:%M')
        formatted_messages.append(f"[{time_str}] {speaker}: {content}")
    
    return "\n".join(formatted_messages)

def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract key phrases from text using simple heuristics
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    if not text:
        return []
    
    # Common customer support keywords
    support_keywords = {
        'problem', 'issue', 'error', 'bug', 'broken', 'not working',
        'help', 'support', 'assistance', 'question', 'confused',
        'refund', 'cancel', 'return', 'exchange', 'billing',
        'account', 'login', 'password', 'access', 'subscription',
        'urgent', 'asap', 'immediately', 'emergency'
    }
    
    # Extract sentences
    sentences = re.split(r'[.!?]+', text)
    phrases = []
    
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if not sentence:
            continue
        
        # Check if sentence contains support keywords
        if any(keyword in sentence for keyword in support_keywords):
            # Clean and add if reasonable length
            if 10 <= len(sentence) <= 100:
                phrases.append(sentence.capitalize())
    
    # Also extract noun phrases (simple approach)
    words = text.lower().split()
    for i, word in enumerate(words):
        if word in support_keywords and i > 0:
            # Get surrounding context
            start = max(0, i - 2)
            end = min(len(words), i + 3)
            phrase = ' '.join(words[start:end])
            if 15 <= len(phrase) <= 80:
                phrases.append(phrase.capitalize())
    
    # Remove duplicates and return top phrases
    unique_phrases = list(dict.fromkeys(phrases))
    return unique_phrases[:max_phrases]

def validate_email(email: str) -> bool:
    """
    Validate email address format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """
    Validate phone number format
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if valid phone format
    """
    if not phone:
        return False
    
    # Remove all non-digit characters
    digits = re.sub(r'[^\d]', '', phone)
    
    # Check if it's a reasonable length (7-15 digits)
    return 7 <= len(digits) <= 15

def get_time_ago(timestamp: datetime) -> str:
    """
    Get human-readable time ago string
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Human-readable time difference
    """
    now = datetime.now()
    if timestamp.tzinfo and not now.tzinfo:
        now = now.replace(tzinfo=timestamp.tzinfo)
    elif not timestamp.tzinfo and now.tzinfo:
        timestamp = timestamp.replace(tzinfo=now.tzinfo)
    
    diff = now - timestamp
    
    if diff.days > 7:
        return timestamp.strftime('%Y-%m-%d')
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"

def create_response_summary(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a summary of response data for analytics
    
    Args:
        response_data: Full response data
        
    Returns:
        Summary dictionary
    """
    metadata = response_data.get('metadata', {})
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'emotion_detected': metadata.get('emotion_detected', 'neutral'),
        'urgency_level': metadata.get('urgency_level', 'low'),
        'escalation_risk': metadata.get('escalation_risk', 'low'),
        'response_strategy': metadata.get('response_strategy', 'standard'),
        'knowledge_sources_used': metadata.get('knowledge_sources_used', 0),
        'tone_adjustments': len(metadata.get('tone_adjustments', [])),
        'response_length': len(response_data.get('response', '')),
        'processing_time': None  # To be filled by caller
    }
    
    return summary

def measure_performance(func):
    """
    Decorator to measure function performance
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.3f} seconds")
        
        # Add timing to result if it's a dictionary
        if isinstance(result, dict):
            result['_execution_time'] = execution_time
        
        return result
    
    return wrapper

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    # Ensure it's not empty
    if not filename:
        filename = "untitled"
    
    return filename

def create_directories(paths: List[str]):
    """
    Create multiple directories if they don't exist
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Directory created/verified: {path}")
        except Exception as e:
            logger.error(f"Error creating directory {path}: {e}")

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0 