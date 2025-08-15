#!/usr/bin/env python3
"""
Local runner for Customer Support RAG System
This script helps set up and run the application locally
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import streamlit
        import chromadb
        import sentence_transformers
        import transformers
        import langchain
        logger.info("✅ All major dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "chroma_db",
        "logs",
        "data",
        "src",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory created/verified: {directory}")

def setup_environment():
    """Set up environment variables"""
    env_file = Path(".env")
    
    if not env_file.exists():
        logger.info("Creating .env file template...")
        with open(env_file, 'w') as f:
            f.write("""# Customer Support RAG Environment Variables
# Copy this file and fill in your actual values

# OpenAI API Key (optional, for enhanced response generation)
OPENAI_API_KEY=your_openai_api_key_here

# Database settings
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Application settings
LOG_LEVEL=INFO
STREAMLIT_SERVER_PORT=8501

# Optional: Hugging Face API token for model downloads
HUGGINGFACE_API_TOKEN=your_hf_token_here
""")
        logger.info("✅ Created .env template file")
    else:
        logger.info("✅ .env file already exists")

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "data/knowledge_base.json",
        "data/sample_conversations.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"⚠️ Missing data files: {missing_files}")
        logger.info("The application will create sample data on first run")
    else:
        logger.info("✅ All data files present")

def run_streamlit():
    """Run the Streamlit application"""
    logger.info("🚀 Starting Customer Support RAG System...")
    
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", ".")
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")

def main():
    """Main function"""
    print("""
    🤖 Customer Support RAG System
    ================================
    
    This script will help you set up and run the Customer Support RAG system locally.
    """)
    
    # Pre-flight checks
    logger.info("Running pre-flight checks...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    logger.info(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Please install requirements first: pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup
    logger.info("Setting up environment...")
    create_directories()
    setup_environment()
    check_data_files()
    
    logger.info("✅ Setup complete!")
    
    # Provide instructions
    print("""
    📋 Quick Start Instructions:
    
    1. (Optional) Add your OpenAI API key to the .env file for enhanced responses
    2. The application will run on http://localhost:8501
    3. Use the sidebar to load sample data and explore features
    4. Try the different tabs: Chat Interface, Analytics, Knowledge Base, Testing
    
    🔧 Configuration:
    - Edit config/config.yaml to customize system behavior
    - Add more knowledge base articles to data/knowledge_base.json
    - Modify response templates in src/response_generator.py
    """)
    
    # Ask user if they want to start
    try:
        start = input("\nStart the application now? (y/n): ").lower().strip()
        if start in ['y', 'yes', '']:
            run_streamlit()
        else:
            logger.info("To start later, run: streamlit run app.py")
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 