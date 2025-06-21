# Cloud Deployment Strategies

## Topics
- Containerization for ML models
- Orchestration with Kubernetes
- Serverless deployment options
- Cloud-based ML serving systems

### Example: Dockerfile for Model Serving
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "serve.py"]
```

### Example: Kubernetes Deployment YAML
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: model-server
        image: myrepo/model-server:latest
        ports:
        - containerPort: 8080
```
