# Deployment Guide

Learn how to deploy Nexus-LLM in production environments using Docker, systemd, and cloud platforms.

---

## Docker Deployment

### Quick Start with Docker Compose

The easiest way to deploy Nexus-LLM is with Docker Compose:

```bash
# Clone and configure
git clone https://github.com/nexus-llm/nexus-llm.git
cd nexus-llm

# Edit configuration
cp .env.example .env
# Edit .env with your API keys and settings

# Start all services
docker compose up -d
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: "3.8"

services:
  nexus-api:
    image: nexusllm/nexus-llm:latest
    container_name: nexus-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config
      - model-cache:/root/.cache/huggingface
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    environment:
      - NEXUS_CONFIG_DIR=/app/config
      - NEXUS_HOST=0.0.0.0
      - NEXUS_PORT=8000
      - NEXUS_LOG_LEVEL=INFO
      - NEXUS_API_KEY=${NEXUS_API_KEY}
      - NEXUS_HF_TOKEN=${NEXUS_HF_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nexus-worker:
    image: nexusllm/nexus-llm:latest
    container_name: nexus-worker
    restart: unless-stopped
    command: ["python", "main.py", "--mode", "worker"]
    volumes:
      - ./config:/app/config
      - model-cache:/root/.cache/huggingface
      - ./checkpoints:/app/checkpoints
    environment:
      - NEXUS_CONFIG_DIR=/app/config
      - NEXUS_HF_TOKEN=${NEXUS_HF_TOKEN}
      - NEXUS_REDIS_URL=redis://redis:6379
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    container_name: nexus-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  nginx:
    image: nginx:alpine
    container_name: nexus-nginx
    restart: unless-stopped
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - nexus-api

volumes:
  model-cache:
  redis-data:
```

### Building a Custom Docker Image

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/cache

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["python3", "main.py", "--mode", "server", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t nexus-llm:latest .
docker run -d --gpus all -p 8000:8000 -v ./config:/app/config nexus-llm:latest
```

---

## Systemd Service

Run Nexus-LLM as a systemd service on Linux for automatic startup and process management.

### Service File

```ini
# /etc/systemd/system/nexus-llm.service
[Unit]
Description=Nexus-LLM API Server
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
User=nexus
Group=nexus
WorkingDirectory=/opt/nexus-llm
Environment="PATH=/opt/nexus-llm/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="NEXUS_CONFIG_DIR=/opt/nexus-llm/config"
Environment="NEXUS_LOG_LEVEL=INFO"
EnvironmentFile=/opt/nexus-llm/.env

ExecStart=/opt/nexus-llm/.venv/bin/python main.py --mode server --host 0.0.0.0 --port 8000
ExecStop=/bin/kill -TERM $MAINPID

Restart=on-failure
RestartSec=10
StartLimitIntervalSec=60
StartLimitBurst=3

# Security hardening
NoNewPrivileges=false
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/opt/nexus-llm/logs /opt/nexus-llm/checkpoints /opt/nexus-llm/cache
PrivateTmp=true

# Resource limits
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### Setup Commands

```bash
# Create system user
sudo useradd -r -s /bin/false -d /opt/nexus-llm nexus

# Set permissions
sudo chown -R nexus:nexus /opt/nexus-llm

# Install and enable service
sudo cp nexus-llm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nexus-llm
sudo systemctl start nexus-llm

# Check status
sudo systemctl status nexus-llm

# View logs
sudo journalctl -u nexus-llm -f
```

---

## Nginx Reverse Proxy

Use Nginx for TLS termination, rate limiting, and load balancing.

```nginx
# /etc/nginx/sites-available/nexus-llm
upstream nexus_api {
    server 127.0.0.1:8000;
    # For multiple workers:
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

# Rate limiting zone
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/m;

server {
    listen 80;
    server_name llm.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name llm.example.com;

    # TLS configuration
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # API endpoints
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://nexus_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running inference
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://nexus_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;
    }

    # Health check (no rate limit)
    location /health {
        proxy_pass http://nexus_api;
    }
}
```

---

## Cloud Deployment

### AWS (EC2 with GPU)

```bash
# Launch a GPU instance (g5.xlarge = 1 A10G GPU, 24GB VRAM)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type g5.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --user-data '#!/bin/bash
    apt-get update
    apt-get install -y python3 python3-pip python3-venv git
    cd /opt
    git clone https://github.com/nexus-llm/nexus-llm.git
    cd nexus-llm
    ./scripts/setup_env.sh
    ./scripts/install.sh
    ./scripts/run.sh --mode server --host 0.0.0.0
  '
```

### Google Cloud (GKE with GPU)

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-llm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nexus-llm
  template:
    metadata:
      labels:
        app: nexus-llm
    spec:
      containers:
      - name: nexus-api
        image: nexusllm/nexus-llm:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: NEXUS_HOST
          value: "0.0.0.0"
        - name: NEXUS_PORT
          value: "8000"
        envFrom:
        - secretRef:
            name: nexus-secrets
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: model-cache
          mountPath: /root/.cache/huggingface
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: config
        configMap:
          name: nexus-config
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-llm-service
spec:
  selector:
    app: nexus-llm
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### RunPod / Vast.ai (GPU Cloud)

For quick GPU access without infrastructure management:

```bash
# SSH into your GPU instance and run:
git clone https://github.com/nexus-llm/nexus-llm.git
cd nexus-llm
./scripts/setup_env.sh --skip-system
./scripts/install.sh
./scripts/download_models.sh mistral-7b
./scripts/run.sh --mode server --host 0.0.0.0
```

---

## Monitoring

### Prometheus Metrics

Nexus-LLM exposes Prometheus metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'nexus-llm'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

Key metrics:
- `nexus_inference_requests_total`
- `nexus_inference_latency_seconds`
- `nexus_tokens_generated_total`
- `nexus_gpu_memory_used_bytes`
- `nexus_gpu_utilization_percent`
- `nexus_model_load_count`

### Grafana Dashboard

Import the included Grafana dashboard:

```bash
# Dashboard JSON is included in monitoring/grafana-dashboard.json
# Import via Grafana UI: Dashboards → Import → Upload JSON file
```

---

## Security Checklist

- [ ] TLS/SSL enabled (via nginx or application config)
- [ ] API authentication enabled (API keys or JWT)
- [ ] CORS origins restricted to your domains
- [ ] Server bound to localhost or internal network (not 0.0.0.0 without a firewall)
- [ ] Firewall rules restrict port 8000 to authorized IPs
- [ ] `.env` file has restricted permissions (`chmod 600 .env`)
- [ ] HuggingFace tokens and API keys are in secrets, not code
- [ ] Rate limiting is configured
- [ ] Log files are rotated and not growing unbounded
- [ ] Regular security updates applied (`./scripts/update.sh`)
