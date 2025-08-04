# CUDA Finance Tool - Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Database Setup](#database-setup)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **Storage**: 10GB+ free space
- **Network**: Internet access for dependencies

### Software Requirements

- **CUDA Toolkit**: 11.8 or later
- **Python**: 3.8 or later
- **Docker**: 20.10+ (for containerized deployment)
- **NVIDIA Docker**: For GPU support in containers

### GPU Requirements

```bash
# Check GPU compatibility
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check CUDA compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Local Development Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/cuda-finance-tool.git
cd cuda-finance-tool

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build the Application

```bash
# Check system compatibility
make setup

# Build the application
make all

# Run tests
make test
```

### 3. Start Development Server

```bash
# Start web interface
make web

# Or run directly
python web_interface/app.py
```

## Docker Deployment

### 1. Basic Docker Setup

```bash
# Build the Docker image
docker build -t cuda-finance-tool .

# Run with GPU support
docker run --gpus all -p 5000:5000 cuda-finance-tool
```

### 2. Docker Compose Setup

```bash
# Start with basic services
docker-compose up -d

# Start with caching (Redis)
docker-compose --profile cache up -d

# Start with database (PostgreSQL)
docker-compose --profile database up -d

# Start with reverse proxy (Nginx)
docker-compose --profile proxy up -d
```

### 3. Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  cuda-finance-tool:
    build: .
    container_name: cuda-finance-tool-prod
    restart: unless-stopped
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/cuda_finance
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: cuda_finance
      POSTGRES_USER: cuda_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - cuda-finance-tool
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

## Production Deployment

### 1. Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cuda-finance-tool
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cuda-finance-tool
  template:
    metadata:
      labels:
        app: cuda-finance-tool
    spec:
      containers:
      - name: cuda-finance-tool
        image: cuda-finance-tool:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cuda-finance-secrets
              key: database-url
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: data
          mountPath: /app/data
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: cuda-finance-logs-pvc
      - name: data
        persistentVolumeClaim:
          claimName: cuda-finance-data-pvc
```

### 2. Environment Variables

Create `.env` file:

```bash
# Application
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DEBUG=False

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/cuda_finance
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your-redis-password

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=/app/logs

# GPU
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_LIMIT=8192

# Security
CORS_ORIGINS=https://your-domain.com
API_RATE_LIMIT=1000
```

### 3. SSL Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream cuda_finance {
        server cuda-finance-tool:5000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://cuda_finance;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## Monitoring and Logging

### 1. Prometheus Metrics

Add to your application:

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage in bytes')
CUDA_KERNEL_DURATION = Histogram('cuda_kernel_duration_seconds', 'CUDA kernel execution time')

# Middleware for metrics
@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint).inc()
    REQUEST_LATENCY.observe(duration)
    return response
```

### 2. Log Aggregation

Configure log shipping to ELK stack or similar:

```yaml
# Filebeat configuration
filebeat.inputs:
- type: log
  paths:
    - /app/logs/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

### 3. Health Checks

```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gpu_available': check_gpu_availability(),
        'database_connected': check_database_connection(),
        'memory_usage': get_memory_usage()
    }
```

## Database Setup

### 1. PostgreSQL Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE cuda_finance;
CREATE USER cuda_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE cuda_finance TO cuda_user;
\q

# Run migrations
alembic upgrade head
```

### 2. Database Migrations

Create `alembic.ini`:

```ini
[alembic]
script_location = migrations
sqlalchemy.url = postgresql://cuda_user:your_password@localhost/cuda_finance
```

### 3. Backup Strategy

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump cuda_finance > backup_$DATE.sql
gzip backup_$DATE.sql
```

## Security Considerations

### 1. Network Security

- Use HTTPS in production
- Implement rate limiting
- Configure firewall rules
- Use VPN for remote access

### 2. Application Security

```python
# Security headers
from flask_talisman import Talisman

Talisman(app, 
    content_security_policy={
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline'",
        'style-src': "'self' 'unsafe-inline'"
    }
)

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

### 3. Data Security

- Encrypt sensitive data at rest
- Use environment variables for secrets
- Implement proper access controls
- Regular security audits

## Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   # Check CUDA installation
   nvcc --version
   nvidia-smi
   
   # Reinstall CUDA if needed
   sudo apt-get install nvidia-cuda-toolkit
   ```

2. **GPU memory errors**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Reduce batch size in config.h
   # DEFAULT_NUM_OPTIONS 4096
   ```

3. **Database connection errors**
   ```bash
   # Check database status
   sudo systemctl status postgresql
   
   # Test connection
   psql -h localhost -U cuda_user -d cuda_finance
   ```

4. **Performance issues**
   ```bash
   # Monitor GPU usage
   watch -n 1 nvidia-smi
   
   # Check application logs
   tail -f logs/cuda_finance.log
   ```

### Performance Tuning

1. **GPU Optimization**
   - Use appropriate block sizes
   - Optimize memory access patterns
   - Use shared memory when possible

2. **Database Optimization**
   - Create indexes on frequently queried columns
   - Use connection pooling
   - Regular VACUUM and ANALYZE

3. **Application Optimization**
   - Use async processing for long-running tasks
   - Implement caching with Redis
   - Monitor and optimize slow queries

### Support and Maintenance

1. **Regular Maintenance**
   - Update dependencies monthly
   - Monitor disk space and logs
   - Backup database regularly
   - Review security patches

2. **Monitoring Setup**
   - Set up alerts for critical metrics
   - Monitor GPU temperature and usage
   - Track application performance
   - Monitor error rates

3. **Documentation**
   - Keep deployment documentation updated
   - Document custom configurations
   - Maintain runbooks for common issues 