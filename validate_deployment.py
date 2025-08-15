#!/usr/bin/env python3
"""
Validation script for Customer Support RAG deployment
Run this to check if all components are working correctly
"""

import os
import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'streamlit',
        'numpy', 
        'pandas',
        'chromadb',
        'sentence_transformers',
        'textblob',
        'vaderSentiment',
        'plotly'
    ]
    
    optional_modules = [
        'transformers',
        'torch',
        'openai',
        'sklearn'
    ]
    
    success = True
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            success = False
    
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} (optional)")
        except ImportError:
            print(f"⚠️  {module} (optional) - not available")
    
    return success

def test_project_structure():
    """Test that all required files exist"""
    print("\n📁 Testing project structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'src/rag_system.py',
        'src/sentiment_analyzer.py',
        'src/escalation_predictor.py',
        'src/response_generator.py',
        'src/utils.py',
        'data/knowledge_base.json',
        'data/sample_conversations.json'
    ]
    
    success = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            success = False
    
    return success

def test_data_files():
    """Test that data files are valid"""
    print("\n📊 Testing data files...")
    
    try:
        import json
        
        # Test knowledge base
        with open('data/knowledge_base.json', 'r') as f:
            kb = json.load(f)
        print(f"✅ Knowledge base: {len(kb)} articles")
        
        # Test sample conversations
        with open('data/sample_conversations.json', 'r') as f:
            convs = json.load(f)
        print(f"✅ Sample conversations: {len(convs)} conversations")
        
        return True
        
    except Exception as e:
        print(f"❌ Data files error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports
        from src.rag_system import RAGSystem
        from src.sentiment_analyzer import SentimentAnalyzer
        print("✅ Module imports successful")
        
        # Test sentiment analyzer
        analyzer = SentimentAnalyzer(use_transformer=False)  # Skip transformer for speed
        result = analyzer.analyze_sentiment("I am happy with your service!")
        print("✅ Sentiment analysis working")
        print(f"   - Sentiment: {result['overall_sentiment']}")
        print(f"   - Confidence: {result['confidence']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Customer Support RAG - Deployment Validation")
    print("=" * 55)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test project structure
    if not test_project_structure():
        all_passed = False
    
    # Test data files
    if not test_data_files():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 55)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your deployment should work correctly")
        print("\n🚀 Ready for Streamlit Cloud deployment!")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Please fix the issues before deploying")
        print("\n💡 Solutions:")
        print("   - Check requirements.txt for missing packages")
        print("   - Verify all files are present")
        print("   - Try using requirements-minimal.txt")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 