#!/usr/bin/env python3
"""
Setup for Deployment Script
Run this in Google Colab to prepare your project for GitHub and Streamlit Cloud deployment
"""

import os
import zipfile
import json
from pathlib import Path

def create_project_structure():
    """Create the complete project structure"""
    
    directories = [
        'src',
        'data', 
        'config',
        '.streamlit',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def validate_files():
    """Validate that all required files exist"""
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'run_local.py',
        'packages.txt',
        '.streamlit/config.toml',
        '.streamlit/secrets.toml.template',
        'src/rag_system.py',
        'src/sentiment_analyzer.py', 
        'src/escalation_predictor.py',
        'src/response_generator.py',
        'src/evaluation.py',
        'src/utils.py',
        'data/knowledge_base.json',
        'data/sample_conversations.json',
        'config/config.yaml',
        'DEPLOYMENT_INSTRUCTIONS.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print(f"\n🎉 All {len(required_files)} required files found!")
        return True

def create_deployment_zip():
    """Create a zip file ready for deployment"""
    
    print("\n📦 Creating deployment zip file...")
    
    with zipfile.ZipFile('customer-support-rag-deployment.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.git') and 
                      d not in ['__pycache__', 'chroma_db', 'logs', '.ipynb_checkpoints']]
            
            for file in files:
                if (not file.startswith('.') and 
                    not file.endswith('.pyc') and 
                    not file.endswith('.zip') and
                    file != 'setup_for_deployment.py'):  # Don't include this script
                    
                    file_path = os.path.join(root, file)
                    # Use relative path in zip
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)
                    print(f"   📄 Added: {arcname}")
    
    print(f"\n✅ Created deployment zip: customer-support-rag-deployment.zip")
    print(f"📊 Zip file size: {os.path.getsize('customer-support-rag-deployment.zip') / 1024 / 1024:.2f} MB")

def generate_deployment_checklist():
    """Generate a deployment checklist"""
    
    checklist = """
# 🚀 Deployment Checklist

## Pre-Deployment
- [ ] All files validated and present
- [ ] Downloaded deployment zip file
- [ ] Have GitHub account ready
- [ ] Have Streamlit Cloud account ready
- [ ] Have OpenAI API key (optional but recommended)

## GitHub Upload
- [ ] Created new public repository on GitHub
- [ ] Uploaded all files from zip
- [ ] Verified all folders and files are present
- [ ] Repository is public (required for free Streamlit Cloud)

## Streamlit Cloud Deployment  
- [ ] Connected GitHub account to Streamlit Cloud
- [ ] Selected correct repository and branch (main)
- [ ] Set main file path to: app.py
- [ ] Added secrets (especially OPENAI_API_KEY)
- [ ] Clicked Deploy and waited for completion

## Testing
- [ ] App loads without errors
- [ ] Knowledge base initializes (10 articles)
- [ ] Sentiment analysis works
- [ ] Response generation works
- [ ] All tabs function properly
- [ ] Analytics dashboard displays

## Final Steps
- [ ] Shared demo URL
- [ ] Shared GitHub repository URL
- [ ] Documented any customizations

## URLs to Share
📝 Demo URL: https://[your-app-name].streamlit.app
📁 GitHub: https://github.com/[your-username]/customer-support-rag

## Support
If you encounter issues:
1. Check DEPLOYMENT_INSTRUCTIONS.md for detailed steps
2. Review Streamlit Cloud logs for specific errors
3. Verify all files uploaded correctly to GitHub
"""
    
    with open('DEPLOYMENT_CHECKLIST.md', 'w') as f:
        f.write(checklist)
    
    print("✅ Created DEPLOYMENT_CHECKLIST.md")

def show_next_steps():
    """Show next steps for deployment"""
    
    print("""
🎯 NEXT STEPS FOR DEPLOYMENT:

1. 📥 DOWNLOAD the zip file: customer-support-rag-deployment.zip
   
2. 📁 CREATE GitHub repository:
   - Go to github.com
   - Create new public repository: 'customer-support-rag'
   - Upload all files from the zip
   
3. ☁️ DEPLOY to Streamlit Cloud:
   - Go to share.streamlit.io
   - Connect your GitHub account
   - Select your repository
   - Set main file: app.py
   - Add your OpenAI API key in secrets
   
4. 🧪 TEST your deployment:
   - Wait for app to load
   - Try the sample conversations
   - Verify all features work

📖 For detailed instructions, see: DEPLOYMENT_INSTRUCTIONS.md

🎉 Your Customer Support RAG system will be live and ready to demo!
""")

def main():
    """Main deployment preparation function"""
    
    print("🤖 Customer Support RAG - Deployment Preparation")
    print("=" * 55)
    
    # Create project structure
    print("\n1. Creating project structure...")
    create_project_structure()
    
    # Validate files
    print("\n2. Validating project files...")
    if not validate_files():
        print("\n❌ Please ensure all files are created before deployment")
        return
    
    # Create deployment zip
    print("\n3. Creating deployment package...")
    create_deployment_zip()
    
    # Generate checklist
    print("\n4. Generating deployment checklist...")
    generate_deployment_checklist()
    
    # Show next steps
    print("\n5. Next steps...")
    show_next_steps()
    
    print("\n✅ Deployment preparation complete!")
    print("📦 Download the zip file and follow the deployment instructions.")

if __name__ == "__main__":
    main() 