# GraphJudge Streamlit Pipeline - Deployment Guide

**Version:** 2.0  
**Last Updated:** 2025-09-13  
**Status:** Production Ready

This guide provides comprehensive instructions for deploying, maintaining, and monitoring the GraphJudge Streamlit Pipeline in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Local Development Deployment](#local-development-deployment)
4. [Production Deployment](#production-deployment)
5. [Configuration Management](#configuration-management)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Backup and Recovery](#backup-and-recovery)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance Procedures](#maintenance-procedures)
10. [Security Considerations](#security-considerations)

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8+ (Recommended: Python 3.11)
- RAM: 2GB minimum, 4GB recommended
- Storage: 1GB free space
- Network: Stable internet connection for API calls

**Recommended Production Environment:**
- Python 3.11
- RAM: 8GB+
- CPU: 2+ cores
- Storage: 10GB+ free space
- Load balancer (for high availability)

### API Keys Required

1. **OpenAI API Key** (or Azure OpenAI)
   - For entity extraction and triple generation
   - Rate limits: Consider Plus/Pro subscription for production
   - Cost estimation: ~$0.01-0.05 per processing request

2. **Perplexity API Key**
   - For graph judgment and reasoning
   - Rate limits: Check current Perplexity limits
   - Cost estimation: ~$0.02-0.10 per judgment request

### Dependencies

Core dependencies are listed in `requirements.txt`:
```bash
streamlit>=1.25.0
litellm>=1.0.0
pydantic>=2.0.0
requests>=2.28.0
python-dotenv>=1.0.0
aiohttp>=3.8.0
```

Development and testing dependencies:
```bash
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
mypy>=1.0.0
flake8>=6.0.0
black>=23.0.0
```

---

## Environment Setup

### 1. Clone and Navigate to Project

```bash
git clone <repository-url>
cd GraphJudge_TextToKG_CLI/streamlit_pipeline
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

### 4. Environment Configuration

Create `.env` file with API credentials:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your API keys
nano .env
```

Required environment variables:
```bash
# Choose ONE of the following OpenAI configurations:

# Option 1: Standard OpenAI
OPENAI_API_KEY=your_openai_key_here

# Option 2: Azure OpenAI (Recommended for enterprise)
AZURE_OPENAI_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Required: Perplexity for graph judgment
PERPLEXITY_API_KEY=your_perplexity_key_here

# Optional: Logging and performance
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_TRACKING=true
CACHE_ENABLED=true
```

---

## Local Development Deployment

### Quick Start

```bash
# Verify installation
python run_tests.py --smoke

# Start development server
streamlit run app.py

# Access application
# Open browser to: http://localhost:8501
```

### Development Configuration

For development, use `.env.development`:
```bash
LOG_LEVEL=DEBUG
ENABLE_DETAILED_LOGGING=true
CACHE_ENABLED=false
API_TIMEOUT=30
MAX_RETRIES=3
```

### Running Tests

```bash
# Full test suite
python run_tests.py --coverage

# Quick smoke tests
python run_tests.py --smoke

# Integration tests
python run_tests.py --integration

# Performance tests
python run_tests.py --performance
```

---

## Production Deployment

### Option 1: Streamlit Cloud (Recommended for Small Scale)

1. **Push code to GitHub repository**
2. **Connect to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository and branch
   - Set app file: `streamlit_pipeline/app.py`

3. **Configure Environment Variables:**
   - In Streamlit Cloud settings, add:
     - `OPENAI_API_KEY` or `AZURE_OPENAI_KEY`
     - `PERPLEXITY_API_KEY`
     - `LOG_LEVEL=INFO`

4. **Deploy and Monitor:**
   - Streamlit Cloud handles deployment automatically
   - Monitor via Streamlit Cloud dashboard

### Option 2: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
# Build image
docker build -t graphjudge-streamlit .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -e PERPLEXITY_API_KEY=your_key \
  graphjudge-streamlit
```

### Option 3: Traditional Server Deployment

#### Using systemd (Linux)

1. **Create service file:** `/etc/systemd/system/graphjudge.service`
```ini
[Unit]
Description=GraphJudge Streamlit Pipeline
After=network.target

[Service]
Type=simple
User=graphjudge
WorkingDirectory=/opt/graphjudge/streamlit_pipeline
Environment=PATH=/opt/graphjudge/venv/bin
ExecStart=/opt/graphjudge/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

2. **Enable and start service:**
```bash
sudo systemctl enable graphjudge.service
sudo systemctl start graphjudge.service
sudo systemctl status graphjudge.service
```

#### Using nginx as Reverse Proxy

Create nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## Configuration Management

### Environment-Specific Configurations

**Development (`config/dev.yaml`):**
```yaml
api:
  timeout: 30
  max_retries: 3
  rate_limit_enabled: false

logging:
  level: DEBUG
  file_logging: true
  detailed_tracing: true

cache:
  enabled: false
  ttl: 300

performance:
  tracking_enabled: true
  profiling_enabled: true
```

**Production (`config/prod.yaml`):**
```yaml
api:
  timeout: 60
  max_retries: 5
  rate_limit_enabled: true

logging:
  level: INFO
  file_logging: true
  detailed_tracing: false

cache:
  enabled: true
  ttl: 3600

performance:
  tracking_enabled: true
  profiling_enabled: false

security:
  api_key_rotation_enabled: true
  request_validation_strict: true
```

### Configuration Loading

The application automatically loads configuration based on environment:

```python
# Automatic environment detection
from streamlit_pipeline.core.config import get_config

config = get_config()  # Loads appropriate config based on ENVIRONMENT variable
```

### Secret Management

**For Production, use proper secret management:**

1. **AWS Secrets Manager:**
```python
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']
```

2. **Azure Key Vault:**
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://vault.vault.azure.net/", credential=credential)
secret = client.get_secret("api-key")
```

3. **Environment Variables (Basic):**
```bash
export OPENAI_API_KEY=$(cat /etc/secrets/openai-key)
export PERPLEXITY_API_KEY=$(cat /etc/secrets/perplexity-key)
```

---

## Monitoring and Logging

### Application Monitoring

**Health Check Endpoint:**
The application provides built-in health checks:
```bash
# Check application health
curl http://localhost:8501/_stcore/health

# Expected response: {"status": "ok"}
```

**Custom Monitoring Dashboard:**
Monitor key metrics:
- Request rate and response time
- API call success rate
- Error rates by type
- Session duration and user engagement
- Resource utilization (CPU, memory)

### Logging Configuration

**Production Logging Setup:**
```python
import logging
from streamlit_pipeline.utils.error_handling import StreamlitLogger

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/graphjudge/app.log'),
        logging.StreamHandler()
    ]
)

# Use application logger
logger = StreamlitLogger()
logger.log_info("Application started", {"version": "2.0", "environment": "production"})
```

**Log Rotation:**
```bash
# Configure logrotate
cat > /etc/logrotate.d/graphjudge << EOF
/var/log/graphjudge/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 graphjudge graphjudge
}
EOF
```

### Performance Monitoring

**Key Metrics to Monitor:**
1. **Response Time:** P50, P95, P99 latencies
2. **Throughput:** Requests per minute
3. **Error Rate:** 4xx and 5xx error percentages
4. **API Costs:** Track OpenAI and Perplexity API usage
5. **Resource Usage:** CPU, memory, disk usage

**Monitoring Tools Integration:**
- **Prometheus + Grafana:** For comprehensive metrics
- **New Relic:** For application performance monitoring
- **DataDog:** For cloud-native monitoring
- **Custom Dashboard:** Built-in application metrics

---

## Backup and Recovery

### Data Backup Strategy

**What to Backup:**
1. **Configuration Files:** `.env`, config files
2. **Session Data:** If persistence is enabled
3. **Logs:** Application and error logs
4. **Custom Models:** Any fine-tuned models

**Backup Script Example:**
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/graphjudge"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup configuration
cp -r /opt/graphjudge/streamlit_pipeline/config $BACKUP_DIR/$DATE/
cp /opt/graphjudge/streamlit_pipeline/.env $BACKUP_DIR/$DATE/

# Backup logs
cp -r /var/log/graphjudge $BACKUP_DIR/$DATE/

# Backup session data (if applicable)
if [ -d "/opt/graphjudge/data" ]; then
    cp -r /opt/graphjudge/data $BACKUP_DIR/$DATE/
fi

# Compress backup
tar -czf $BACKUP_DIR/graphjudge_backup_$DATE.tar.gz -C $BACKUP_DIR $DATE
rm -rf $BACKUP_DIR/$DATE

echo "Backup completed: $BACKUP_DIR/graphjudge_backup_$DATE.tar.gz"
```

### Recovery Procedures

**Quick Recovery Steps:**
1. **Stop Application:**
   ```bash
   sudo systemctl stop graphjudge
   ```

2. **Restore from Backup:**
   ```bash
   cd /opt/graphjudge
   tar -xzf /backups/graphjudge/graphjudge_backup_YYYYMMDD_HHMMSS.tar.gz
   ```

3. **Verify Configuration:**
   ```bash
   cd streamlit_pipeline
   python run_tests.py --smoke
   ```

4. **Restart Application:**
   ```bash
   sudo systemctl start graphjudge
   ```

### Disaster Recovery Plan

**RTO (Recovery Time Objective):** 15 minutes  
**RPO (Recovery Point Objective):** 1 hour

**Recovery Steps:**
1. **Infrastructure Recovery:** Provision new server/container
2. **Code Deployment:** Deploy from version control
3. **Configuration Restore:** Restore API keys and settings
4. **Data Recovery:** Restore any persistent data
5. **Verification:** Run health checks and smoke tests
6. **DNS Update:** Point traffic to new instance

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Application Won't Start

**Symptoms:**
- Import errors
- Module not found errors
- Configuration errors

**Solutions:**
```bash
# Check Python environment
python --version
pip list | grep streamlit

# Verify requirements
pip install -r requirements.txt

# Check configuration
python -c "from streamlit_pipeline.core.config import get_config; print(get_config())"

# Run smoke tests
python run_tests.py --smoke
```

#### 2. API Authentication Errors

**Symptoms:**
- 401 Unauthorized errors
- Invalid API key messages

**Solutions:**
```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $PERPLEXITY_API_KEY

# Test API connectivity
python -c "
from streamlit_pipeline.utils.api_client import get_api_client
client = get_api_client()
print('API client created successfully')
"

# Check API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### 3. Performance Issues

**Symptoms:**
- Slow response times
- High CPU/memory usage
- API timeouts

**Solutions:**
```bash
# Monitor resource usage
htop
df -h

# Check API rate limits
# Review logs for rate limit errors

# Optimize configuration
# Increase timeouts in production config
# Enable caching
# Implement request queuing
```

#### 4. Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of memory errors

**Solutions:**
```bash
# Monitor memory usage
ps aux | grep streamlit
free -h

# Clear session state
# Restart application periodically
# Review session cleanup configuration
```

### Debug Mode

Enable detailed debugging:
```bash
export LOG_LEVEL=DEBUG
export ENABLE_DETAILED_LOGGING=true
streamlit run app.py --logger.level=debug
```

### Log Analysis

**Useful log analysis commands:**
```bash
# Recent errors
grep ERROR /var/log/graphjudge/app.log | tail -20

# API call patterns
grep "API call" /var/log/graphjudge/app.log | awk '{print $1, $2}' | sort | uniq -c

# Performance metrics
grep "processing_time" /var/log/graphjudge/app.log | awk '{print $NF}' | sort -n
```

---

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily Tasks
- [ ] Monitor application health and performance
- [ ] Check error logs for unusual patterns
- [ ] Verify API quota usage
- [ ] Monitor disk space and memory usage

#### Weekly Tasks
- [ ] Review performance metrics and trends
- [ ] Update dependencies (security patches)
- [ ] Clean up old logs and temporary files
- [ ] Backup configuration and data

#### Monthly Tasks
- [ ] Full system backup
- [ ] Performance optimization review
- [ ] Security audit and vulnerability scan
- [ ] Dependency updates and testing
- [ ] Capacity planning review

### Update Procedures

#### Application Updates

1. **Prepare Update:**
   ```bash
   # Create backup
   ./backup.sh
   
   # Test update in staging
   git checkout staging
   python run_tests.py --all
   ```

2. **Deploy Update:**
   ```bash
   # Stop application
   sudo systemctl stop graphjudge
   
   # Update code
   git pull origin main
   
   # Update dependencies
   pip install -r requirements.txt
   
   # Run tests
   python run_tests.py --smoke
   
   # Restart application
   sudo systemctl start graphjudge
   ```

3. **Verify Update:**
   ```bash
   # Check application health
   curl http://localhost:8501/_stcore/health
   
   # Monitor logs
   tail -f /var/log/graphjudge/app.log
   ```

#### Dependency Updates

```bash
# Check for outdated packages
pip list --outdated

# Update packages (careful with major version changes)
pip install --upgrade package_name

# Test thoroughly after updates
python run_tests.py --all
```

### Performance Optimization

#### Regular Optimization Tasks

1. **Cache Optimization:**
   - Review cache hit rates
   - Adjust cache TTL settings
   - Clear stale cache entries

2. **Session Management:**
   - Configure session cleanup
   - Monitor session state size
   - Implement session timeout

3. **API Optimization:**
   - Monitor API response times
   - Implement request batching where possible
   - Optimize prompt engineering

### Capacity Planning

**Monitor Key Metrics:**
- Concurrent users
- Request rate
- Resource utilization
- API costs

**Scaling Indicators:**
- CPU usage consistently > 70%
- Memory usage > 80%
- Response time > 5 seconds
- Error rate > 5%

**Scaling Options:**
- Vertical scaling (increase server resources)
- Horizontal scaling (multiple instances + load balancer)
- API optimization and caching
- CDN implementation for static assets

---

## Security Considerations

### Security Best Practices

#### 1. API Key Management
- **Never commit API keys to version control**
- **Use environment variables or secret management systems**
- **Rotate API keys regularly**
- **Monitor API key usage for anomalies**

#### 2. Network Security
- **Use HTTPS in production**
- **Implement proper firewall rules**
- **Consider VPN access for internal tools**
- **Rate limiting and DDoS protection**

#### 3. Application Security
- **Input validation and sanitization**
- **Output encoding**
- **Session security**
- **Regular security audits**

### Security Monitoring

**Security Events to Monitor:**
- Failed authentication attempts
- Unusual API usage patterns
- Error spikes (potential attacks)
- Configuration changes

**Security Tools:**
```bash
# Vulnerability scanning
bandit -r core/ utils/
safety check

# Dependency security audit
pip-audit

# SSL/TLS testing
curl -I https://your-domain.com
```

### Incident Response Plan

**Security Incident Response:**
1. **Detect:** Monitor logs and alerts
2. **Contain:** Isolate affected systems
3. **Investigate:** Analyze logs and evidence
4. **Remediate:** Fix vulnerabilities
5. **Recover:** Restore normal operations
6. **Learn:** Update procedures and monitoring

---

## Support and Contacts

### Getting Help

**Documentation:**
- README.md - Quick start guide
- API_Integration.md - API configuration details
- TASK.md - Development progress tracking

**Testing:**
```bash
# Run comprehensive tests
python run_tests.py --all

# Get test coverage report
python run_tests.py --html-coverage
```

**Community:**
- GitHub Issues: Report bugs and feature requests
- Development Team: For urgent production issues

### Emergency Contacts

**Production Issues:**
- Primary Contact: Development Team Lead
- Secondary Contact: DevOps Team
- Escalation: Technical Director

**API Provider Issues:**
- OpenAI Support: Platform support
- Perplexity Support: API support

---

**Last Updated:** 2025-09-13  
**Next Review:** Monthly  
**Version:** 2.0