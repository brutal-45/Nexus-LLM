# Deployment Guide

This guide covers deploying Nexus-LLM in production environments using Docker, Kubernetes, and cloud platforms.

## Docker Deployment

### Quick Start

```bash
# Build the image
docker build -t nexus-llm:2.1.0 .

# Run with GPU support
docker run --gpus all -p 8000:8000 \
  -v ./models:/app/models \
  -v ./config:/app/config \
  nexus-llm:2.1.0
```

### Docker Compose

```yaml
version: '3.8'

services:
  nexus-llm:
    image: nexus-llm:2.1.0
    ports:
      - "8000:8000"
      - "9090:9090"    # Prometheus metrics
    volumes:
      - ./models:/app/models
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - NEXUS_CONFIG=/app/config/production.yaml
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

### Configuration

Create `config/production.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins: ["https://your-app.example.com"]
  request_timeout: 120
  max_concurrent_requests: 100

models:
  - id: nexus-7b-chat
    path: /app/models/nexus-7b-chat
    device: auto
    dtype: float16
    max_batch_size: 16

auth:
  enabled: true
  jwt:
    secret_key: ${JWT_SECRET}
    access_token_expire_minutes: 60
  api_key:
    prefix: "nxs-"
    hash_keys: true

monitoring:
  prometheus:
    enabled: true
    port: 9090
  logging:
    level: info
    format: json
```

## Kubernetes Deployment

### Namespace and Secrets

```bash
kubectl create namespace nexus-llm
kubectl create secret generic nexus-secrets \
  --from-literal=jwt-secret=$(openssl rand -hex 32) \
  -n nexus-llm
```

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-llm
  namespace: nexus-llm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nexus-llm
  template:
    metadata:
      labels:
        app: nexus-llm
    spec:
      containers:
      - name: nexus-llm
        image: nexus-llm:2.1.0
        ports:
        - containerPort: 8000
        - containerPort: 9090
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "24Gi"
        env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: nexus-secrets
              key: jwt-secret
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: config
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: nexus-models-pvc
      - name: config
        configMap:
          name: nexus-config
      nodeSelector:
        accelerator: nvidia-a100
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nexus-llm-hpa
  namespace: nexus-llm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nexus-llm
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu-utilization
      target:
        type: AverageValue
        averageValue: "70"
```

### Service and Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nexus-llm-service
  namespace: nexus-llm
spec:
  selector:
    app: nexus-llm
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nexus-llm-ingress
  namespace: nexus-llm
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.nexus-llm.example.com
    secretName: nexus-llm-tls
  rules:
  - host: api.nexus-llm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nexus-llm-service
            port:
              number: 8000
```

## Cloud Deployments

### AWS (EKS + EC2 GPU)

1. Create an EKS cluster with GPU node groups (g5.xlarge or p4d.24xlarge)
2. Install the NVIDIA device plugin
3. Apply the Kubernetes manifests above
4. Use EFS for model storage (shared across pods)

### GCP (GKE + A100)

1. Create a GKE cluster with A100 node pools
2. Install GPU drivers via DaemonSet
3. Use Filestore for persistent model storage
4. Configure Workload Identity for secret access

### Azure (AKS + NCas T4)

1. Create an AKS cluster with NCas_T4_v3 series nodes
2. Install the NVIDIA device plugin
3. Use Azure Files for model storage
4. Configure Azure Key Vault for secrets

## Scaling Strategies

### Vertical Scaling
- Increase GPU type (T4 → A100 → H100)
- Add more VRAM for larger batch sizes and models

### Horizontal Scaling
- Add replicas behind a load balancer
- Use HPA based on GPU utilization and request queue length
- Minimum 2 replicas for high availability

### Model Sharding
For models larger than a single GPU:

```yaml
models:
  - id: nexus-70b-chat
    path: /app/models/nexus-70b-chat
    tensor_parallel_size: 4    # Split across 4 GPUs
```

## Monitoring in Production

- **Prometheus**: Scrape `/metrics` on port 9090
- **Grafana**: Use the provided dashboard template in `monitoring/grafana/`
- **Alerts**: Configure alerts for latency, error rate, and GPU memory
- **Logs**: Use structured JSON logging with `log_level: info`

## Security Checklist

- [ ] Enable HTTPS via TLS certificates
- [ ] Require authentication on all endpoints
- [ ] Use environment variables for secrets (not config files)
- [ ] Restrict CORS to your application domains
- [ ] Enable rate limiting
- [ ] Run as non-root user in containers
- [ ] Use network policies to restrict pod communication
- [ ] Rotate API keys and JWT secrets regularly
