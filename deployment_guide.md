# Deployment Guide - Customer Support RAG System

This guide covers deployment options for the Customer Support RAG system across different platforms.

## 🚀 Quick Start - Local Development

1. **Clone and Setup**
```bash
git clone https://github.com/yourusername/customer-support-rag.git
cd customer-support-rag
pip install -r requirements.txt
python run_local.py
```

2. **Access Application**
- Open browser to `http://localhost:8501`
- The system will initialize automatically on first run

## ☁️ Streamlit Cloud Deployment (Recommended)

### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account (free at share.streamlit.io)

### Steps

1. **Prepare Repository**
   - Ensure all code is pushed to GitHub
   - Verify `requirements.txt` is complete
   - Add secrets for sensitive data (see below)

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets**
   In Streamlit Cloud dashboard, add these secrets:
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "your-openai-api-key"
   CHROMA_PERSIST_DIRECTORY = "./chroma_db"
   LOG_LEVEL = "INFO"
   ```

4. **Custom Domain (Optional)**
   - Configure custom domain in Streamlit Cloud settings
   - Update DNS records as instructed

### Streamlit Cloud Benefits
- ✅ Free hosting for public repos
- ✅ Automatic deployments on git push
- ✅ Built-in secrets management
- ✅ No server management required

## 🐳 Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db logs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Commands
```bash
# Build image
docker build -t customer-support-rag .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-api-key \
  -v $(pwd)/chroma_db:/app/chroma_db \
  customer-support-rag

# Run with docker-compose
docker-compose up -d
```

### Docker Compose File
```yaml
version: '3.8'

services:
  customer-support-rag:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## ☁️ Cloud Platform Deployments

### AWS (Amazon Web Services)

#### EC2 Deployment
```bash
# 1. Launch EC2 instance (t2.medium or larger recommended)
# 2. Connect to instance
ssh -i your-key.pem ec2-user@your-instance-ip

# 3. Install dependencies
sudo yum update -y
sudo yum install -y python3 git
pip3 install -r requirements.txt

# 4. Clone repository
git clone https://github.com/yourusername/customer-support-rag.git
cd customer-support-rag

# 5. Set environment variables
export OPENAI_API_KEY="your-api-key"

# 6. Run application
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

#### ECS (Elastic Container Service)
```bash
# 1. Build and push to ECR
aws ecr create-repository --repository-name customer-support-rag
docker build -t customer-support-rag .
docker tag customer-support-rag:latest AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/customer-support-rag:latest
docker push AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/customer-support-rag:latest

# 2. Create ECS task definition and service
# Use AWS Console or CLI to create ECS resources
```

### Google Cloud Platform (GCP)

#### Cloud Run Deployment
```bash
# 1. Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# 2. Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/customer-support-rag
gcloud run deploy customer-support-rag \
  --image gcr.io/PROJECT-ID/customer-support-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --set-env-vars OPENAI_API_KEY=your-api-key
```

#### Compute Engine
```bash
# Similar to EC2 deployment
# Use startup script for automatic setup
```

### Microsoft Azure

#### Container Instances
```bash
# 1. Create resource group
az group create --name customer-support-rg --location eastus

# 2. Deploy container
az container create \
  --resource-group customer-support-rg \
  --name customer-support-rag \
  --image your-docker-hub/customer-support-rag \
  --ports 8501 \
  --environment-variables OPENAI_API_KEY=your-api-key \
  --restart-policy Always
```

## 🔧 Environment Configuration

### Required Environment Variables
```bash
# Essential
OPENAI_API_KEY=your-openai-api-key          # For enhanced responses
CHROMA_PERSIST_DIRECTORY=./chroma_db        # Vector database storage

# Optional
LOG_LEVEL=INFO                              # Logging level
STREAMLIT_SERVER_PORT=8501                  # Port number
HUGGINGFACE_API_TOKEN=your-hf-token         # For model downloads
```

### Secrets Management

#### Streamlit Cloud
- Use `.streamlit/secrets.toml` file
- Add secrets via Streamlit Cloud dashboard

#### Docker
- Use environment variables
- Or mount secrets as files

#### Cloud Platforms
- AWS: Secrets Manager, Parameter Store
- GCP: Secret Manager
- Azure: Key Vault

## 📊 Performance Optimization

### For Production Deployment

1. **Resource Requirements**
   - **CPU**: 2+ cores recommended
   - **RAM**: 4GB+ for smooth operation
   - **Storage**: 10GB+ for models and data
   - **Network**: Stable internet for model downloads

2. **Caching Strategy**
   ```python
   # Enable caching in production
   @st.cache_resource
   def load_models():
       # Model loading logic
       pass
   ```

3. **Database Optimization**
   - Use persistent vector database storage
   - Consider external vector databases for scale
   - Implement proper backup strategies

4. **Monitoring**
   - Set up application monitoring
   - Monitor resource usage
   - Implement health checks

## 🔐 Security Best Practices

### API Keys and Secrets
- Never commit API keys to repository
- Use environment variables or secret management
- Rotate keys regularly

### Network Security
- Use HTTPS in production
- Implement proper firewall rules
- Consider VPN for sensitive deployments

### Data Privacy
- Anonymize customer data
- Implement data retention policies
- Ensure GDPR compliance if applicable

## 🚨 Troubleshooting

### Common Issues

#### Model Download Failures
```bash
# Solution: Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### Memory Issues
```bash
# Solution: Increase memory or use smaller models
# Edit config/config.yaml to use lighter models
```

#### Port Conflicts
```bash
# Solution: Change port in deployment
streamlit run app.py --server.port=8502
```

#### Database Connection Issues
```bash
# Solution: Check permissions and paths
ls -la chroma_db/
chmod -R 755 chroma_db/
```

### Health Checks

#### Basic Health Check
```python
# Add to app.py for monitoring
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

#### Advanced Monitoring
- Use Streamlit's built-in health endpoint: `/_stcore/health`
- Implement custom metrics collection
- Set up alerting for failures

## 📈 Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use shared vector database (Pinecone, Weaviate)
- Implement proper session management

### Vertical Scaling
- Increase instance resources
- Optimize model loading and caching
- Profile application performance

### Database Scaling
- Migrate from ChromaDB to cloud vector databases
- Implement database clustering
- Use read replicas for better performance

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python -m pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Trigger Streamlit Cloud deployment
      run: echo "Deployment triggered automatically by Streamlit Cloud"
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks
- Update dependencies monthly
- Monitor model performance
- Backup vector database
- Review and update knowledge base
- Check security vulnerabilities

### Getting Help
- Check GitHub Issues for common problems
- Review Streamlit documentation
- Contact support team for enterprise deployments

---

**Note**: Replace placeholder values (API keys, repository URLs, etc.) with your actual values before deployment. 