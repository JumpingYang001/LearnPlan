# Project 5: Secure Microservices Architecture

# Project 5: Secure Microservices Architecture

*Duration: 3-4 weeks | Difficulty: Advanced | Prerequisites: Basic microservices knowledge, OAuth2/JWT understanding*

## Project Overview

This comprehensive project guides you through building a production-ready secure microservices architecture with modern authentication, authorization, and monitoring capabilities. You'll implement OAuth2/OpenID Connect authentication, service-to-service security, and comprehensive security monitoring.

### What You'll Build

A complete e-commerce microservices system with:
- **API Gateway** with centralized authentication
- **User Service** with OAuth2/OpenID Connect
- **Product Service** with role-based access control
- **Order Service** with service-to-service authentication
- **Security Monitoring Service** with audit logging
- **Identity Provider** (Keycloak integration)

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Mobile App     │    │  Admin Panel    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      API Gateway          │
                    │   (Authentication &       │
                    │    Rate Limiting)         │
                    └─────────────┬─────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                       │                       │
┌───────▼────────┐    ┌─────────▼────────┐    ┌─────────▼────────┐
│  User Service  │    │ Product Service  │    │  Order Service   │
│   (OAuth2)     │    │    (RBAC)        │    │   (JWT Auth)     │
└───────┬────────┘    └─────────┬────────┘    └─────────┬────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Security Monitoring │
                    │   & Audit Service   │
                    └─────────────────────┘
```

## Description

### Core Features to Implement

1. **OAuth2/OpenID Connect Authentication**
   - User registration and login
   - Token-based authentication
   - Refresh token mechanism
   - Social login integration

2. **Service-to-Service Authentication**
   - JWT-based inter-service communication
   - Service identity management
   - API key authentication for internal services

3. **Security Monitoring and Auditing**
   - Request/response logging
   - Security event monitoring
   - Audit trail generation
   - Anomaly detection

4. **Role-Based Access Control (RBAC)**
   - User roles and permissions
   - Resource-based authorization
   - Dynamic permission checking

### Technologies Used

- **Backend**: Python (FastAPI), Node.js (Express), or Spring Boot
- **Authentication**: Keycloak, Auth0, or custom OAuth2 server
- **API Gateway**: Kong, Ambassador, or custom
- **Monitoring**: ELK Stack, Prometheus + Grafana
- **Databases**: PostgreSQL, Redis
- **Message Queue**: RabbitMQ or Apache Kafka
- **Containerization**: Docker, Docker Compose

## Complete Implementation Guide

### Phase 1: Project Setup and Infrastructure

#### 1.1 Project Structure
```
secure-microservices/
├── docker-compose.yml
├── .env
├── api-gateway/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
├── services/
│   ├── user-service/
│   ├── product-service/
│   ├── order-service/
│   └── security-service/
├── keycloak/
│   └── realm-config.json
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   └── elasticsearch/
└── scripts/
    ├── setup.sh
    └── deploy.sh
```

#### 1.2 Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Identity Provider
  keycloak:
    image: quay.io/keycloak/keycloak:latest
    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin123
    ports:
      - "8080:8080"
    command: start-dev
    volumes:
      - ./keycloak/realm-config.json:/opt/keycloak/data/import/realm.json

  # Databases
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: microservices_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  # Message Queue
  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin123
    ports:
      - "5672:5672"
      - "15672:15672"

  # API Gateway
  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    environment:
      - KEYCLOAK_URL=http://keycloak:8080
      - REDIS_URL=redis://redis:6379
    depends_on:
      - keycloak
      - redis

  # Microservices
  user-service:
    build: ./services/user-service
    ports:
      - "8001:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres123@postgres:5432/microservices_db
      - KEYCLOAK_URL=http://keycloak:8080
    depends_on:
      - postgres
      - keycloak

  product-service:
    build: ./services/product-service
    ports:
      - "8002:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres123@postgres:5432/microservices_db
    depends_on:
      - postgres

  order-service:
    build: ./services/order-service
    ports:
      - "8003:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres123@postgres:5432/microservices_db
      - RABBITMQ_URL=amqp://admin:admin123@rabbitmq:5672
    depends_on:
      - postgres
      - rabbitmq

  security-service:
    build: ./services/security-service
    ports:
      - "8004:8000"
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  # Monitoring Stack
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  elasticsearch_data:
  grafana_data:
```

### Phase 2: API Gateway Implementation

#### 2.1 API Gateway with Authentication
```python
# api-gateway/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import httpx
import jwt
import redis
import json
import time
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Secure API Gateway", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Redis client for caching and rate limiting
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Keycloak configuration
KEYCLOAK_URL = "http://keycloak:8080"
KEYCLOAK_REALM = "microservices"
KEYCLOAK_CLIENT_ID = "api-gateway"

# Service registry
SERVICES = {
    "user": "http://user-service:8000",
    "product": "http://product-service:8000",
    "order": "http://order-service:8000",
    "security": "http://security-service:8000"
}

class SecurityMiddleware:
    """Custom security middleware for authentication and authorization"""
    
    @staticmethod
    async def verify_token(credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Verify JWT token with Keycloak"""
        try:
            token = credentials.credentials
            
            # Check if token is cached
            cached_user = redis_client.get(f"token:{token}")
            if cached_user:
                return json.loads(cached_user)
            
            # Verify token with Keycloak
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid_connect/userinfo",
                    headers={"Authorization": f"Bearer {token}"}
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                user_info = response.json()
                
                # Cache user info for 5 minutes
                redis_client.setex(f"token:{token}", 300, json.dumps(user_info))
                
                return user_info
                
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(status_code=401, detail="Token verification failed")

class RateLimiter:
    """Rate limiting implementation"""
    
    @staticmethod
    async def check_rate_limit(request: Request) -> bool:
        """Check if request is within rate limits"""
        client_ip = request.client.host
        current_time = int(time.time())
        window = 60  # 1 minute window
        limit = 100  # 100 requests per minute
        
        key = f"rate_limit:{client_ip}:{current_time // window}"
        current_requests = redis_client.incr(key)
        redis_client.expire(key, window)
        
        if current_requests > limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        return True

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    return await SecurityMiddleware.verify_token(credentials)

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for all requests"""
    start_time = time.time()
    
    # Rate limiting
    await RateLimiter.check_rate_limit(request)
    
    # Process request
    response = await call_next(request)
    
    # Log request for security monitoring
    process_time = time.time() - start_time
    await log_security_event(request, response, process_time)
    
    return response

async def log_security_event(request: Request, response: Response, process_time: float):
    """Log security events for monitoring"""
    event = {
        "timestamp": time.time(),
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host,
        "status_code": response.status_code,
        "process_time": process_time,
        "user_agent": request.headers.get("user-agent", ""),
    }
    
    # Add authentication info if available
    auth_header = request.headers.get("authorization")
    if auth_header:
        try:
            token = auth_header.split(" ")[1]
            user_info = json.loads(redis_client.get(f"token:{token}") or "{}")
            event["user_id"] = user_info.get("sub", "")
            event["username"] = user_info.get("preferred_username", "")
        except:
            pass
    
    # Send to security service
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{SERVICES['security']}/events",
                json=event,
                timeout=1.0
            )
    except:
        logger.warning("Failed to send security event")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Authentication endpoints
@app.post("/auth/login")
async def login(credentials: dict):
    """Login endpoint - proxy to Keycloak"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid_connect/token",
                data={
                    "grant_type": "password",
                    "client_id": KEYCLOAK_CLIENT_ID,
                    "username": credentials["username"],
                    "password": credentials["password"]
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            return response.json()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/refresh")
async def refresh_token(token_data: dict):
    """Refresh token endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid_connect/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": KEYCLOAK_CLIENT_ID,
                    "refresh_token": token_data["refresh_token"]
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid refresh token")
            
            return response.json()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

# Service proxy endpoints
@app.api_route("/api/users/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def user_service_proxy(
    path: str,
    request: Request,
    current_user = Depends(get_current_user)
):
    """Proxy requests to user service"""
    return await proxy_request("user", path, request)

@app.api_route("/api/products/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def product_service_proxy(
    path: str,
    request: Request,
    current_user = Depends(get_current_user)
):
    """Proxy requests to product service"""
    return await proxy_request("product", path, request, current_user)

@app.api_route("/api/orders/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def order_service_proxy(
    path: str,
    request: Request,
    current_user = Depends(get_current_user)
):
    """Proxy requests to order service"""
    return await proxy_request("order", path, request, current_user)

async def proxy_request(service: str, path: str, request: Request, user_context=None):
    """Generic request proxy function"""
    try:
        url = f"{SERVICES[service]}/{path}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers.pop("host", None)  # Remove host header
        
        # Add user context for service-to-service calls
        if user_context:
            headers["X-User-ID"] = user_context.get("sub", "")
            headers["X-User-Roles"] = ",".join(user_context.get("realm_access", {}).get("roles", []))
        
        # Prepare request data
        body = await request.body()
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
                params=dict(request.query_params)
            )
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
    except Exception as e:
        logger.error(f"Proxy request failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Phase 3: User Service Implementation

#### 3.1 User Service with OAuth2 Integration
```python
# services/user-service/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import httpx
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="User Service", version="1.0.0")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres123@localhost:5432/microservices_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    keycloak_id = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    first_name: str
    last_name: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    keycloak_id: str
    username: str
    email: str
    first_name: str
    last_name: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Security dependencies
security = HTTPBearer()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user_id(x_user_id: str = Header(..., alias="X-User-ID")):
    """Get current user ID from gateway headers"""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID not provided")
    return x_user_id

async def get_user_roles(x_user_roles: str = Header("", alias="X-User-Roles")):
    """Get user roles from gateway headers"""
    return x_user_roles.split(",") if x_user_roles else []

class KeycloakService:
    """Service for interacting with Keycloak"""
    
    def __init__(self):
        self.keycloak_url = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
        self.realm = "microservices"
        self.admin_client_id = "admin-cli"
    
    async def get_admin_token(self):
        """Get admin token for Keycloak operations"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.keycloak_url}/realms/master/protocol/openid_connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.admin_client_id,
                    "username": "admin",
                    "password": "admin123"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to get admin token")
            
            return response.json()["access_token"]
    
    async def create_user(self, user_data: UserCreate):
        """Create user in Keycloak"""
        admin_token = await self.get_admin_token()
        
        keycloak_user = {
            "username": user_data.username,
            "email": user_data.email,
            "firstName": user_data.first_name,
            "lastName": user_data.last_name,
            "enabled": True,
            "credentials": [{
                "type": "password",
                "value": user_data.password,
                "temporary": False
            }]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.keycloak_url}/admin/realms/{self.realm}/users",
                json=keycloak_user,
                headers={"Authorization": f"Bearer {admin_token}"}
            )
            
            if response.status_code not in [201, 409]:  # 409 = user already exists
                raise HTTPException(status_code=400, detail="Failed to create user in Keycloak")
            
            # Get the created user ID
            location = response.headers.get("Location")
            if location:
                return location.split("/")[-1]
            else:
                # User already exists, get ID
                users_response = await client.get(
                    f"{self.keycloak_url}/admin/realms/{self.realm}/users",
                    params={"username": user_data.username},
                    headers={"Authorization": f"Bearer {admin_token}"}
                )
                users = users_response.json()
                if users:
                    return users[0]["id"]
                else:
                    raise HTTPException(status_code=500, detail="User creation failed")
    
    async def update_user(self, keycloak_id: str, user_data: UserUpdate):
        """Update user in Keycloak"""
        admin_token = await self.get_admin_token()
        
        update_data = {}
        if user_data.email:
            update_data["email"] = user_data.email
        if user_data.first_name:
            update_data["firstName"] = user_data.first_name
        if user_data.last_name:
            update_data["lastName"] = user_data.last_name
        
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.keycloak_url}/admin/realms/{self.realm}/users/{keycloak_id}",
                json=update_data,
                headers={"Authorization": f"Bearer {admin_token}"}
            )
            
            if response.status_code != 204:
                raise HTTPException(status_code=400, detail="Failed to update user in Keycloak")
    
    async def delete_user(self, keycloak_id: str):
        """Delete user from Keycloak"""
        admin_token = await self.get_admin_token()
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.keycloak_url}/admin/realms/{self.realm}/users/{keycloak_id}",
                headers={"Authorization": f"Bearer {admin_token}"}
            )
            
            if response.status_code != 204:
                raise HTTPException(status_code=400, detail="Failed to delete user from Keycloak")

keycloak_service = KeycloakService()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "user-service"}

@app.post("/users/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    try:
        # Create user in Keycloak
        keycloak_id = await keycloak_service.create_user(user_data)
        
        # Create user in local database
        db_user = User(
            keycloak_id=keycloak_id,
            username=user_data.username,
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"User registered: {user_data.username}")
        return db_user
        
    except Exception as e:
        db.rollback()
        logger.error(f"User registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="User registration failed")

@app.get("/users/me", response_model=UserResponse)
async def get_current_user(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Get current user profile"""
    user = db.query(User).filter(User.keycloak_id == current_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.put("/users/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    user = db.query(User).filter(User.keycloak_id == current_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Update in Keycloak
        await keycloak_service.update_user(user.keycloak_id, user_data)
        
        # Update in local database
        if user_data.email:
            user.email = user_data.email
        if user_data.first_name:
            user.first_name = user_data.first_name
        if user_data.last_name:
            user.last_name = user_data.last_name
        
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
        
        logger.info(f"User updated: {user.username}")
        return user
        
    except Exception as e:
        db.rollback()
        logger.error(f"User update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="User update failed")

@app.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """List users (admin only)"""
    if "admin" not in user_roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    if "admin" not in user_roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Delete from Keycloak
        await keycloak_service.delete_user(user.keycloak_id)
        
        # Delete from local database
        db.delete(user)
        db.commit()
        
        logger.info(f"User deleted: {user.username}")
        return {"message": "User deleted successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"User deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="User deletion failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 3.2 User Service Dockerfile
```dockerfile
# services/user-service/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 3.3 User Service Requirements
```txt
# services/user-service/requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
pydantic[email]==2.5.0
httpx==0.25.2
python-multipart==0.0.6
```

### Phase 4: Product Service with RBAC

#### 4.1 Product Service Implementation
```python
# services/product-service/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from sqlalchemy import create_engine, Column, Integer, String, Text, DECIMAL, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import Optional, List
from decimal import Decimal
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Product Service", version="1.0.0")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres123@localhost:5432/microservices_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Product model
class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    price = Column(DECIMAL(10, 2))
    sku = Column(String, unique=True, index=True)
    category = Column(String, index=True)
    stock_quantity = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class ProductCreate(BaseModel):
    name: str
    description: str
    price: Decimal
    sku: str
    category: str
    stock_quantity: int = 0

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[Decimal] = None
    category: Optional[str] = None
    stock_quantity: Optional[int] = None
    is_active: Optional[bool] = None

class ProductResponse(BaseModel):
    id: int
    name: str
    description: str
    price: Decimal
    sku: str
    category: str
    stock_quantity: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Dependencies
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user_id(x_user_id: str = Header(None, alias="X-User-ID")):
    """Get current user ID from gateway headers"""
    return x_user_id

async def get_user_roles(x_user_roles: str = Header("", alias="X-User-Roles")):
    """Get user roles from gateway headers"""
    return x_user_roles.split(",") if x_user_roles else []

class PermissionChecker:
    """Role-based access control implementation"""
    
    @staticmethod
    def check_permission(required_roles: List[str], user_roles: List[str]):
        """Check if user has required permissions"""
        if not user_roles:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if "admin" in user_roles:
            return True  # Admin has all permissions
        
        for role in required_roles:
            if role in user_roles:
                return True
        
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    @staticmethod
    def require_roles(*roles):
        """Decorator for role-based access control"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                user_roles = kwargs.get('user_roles', [])
                PermissionChecker.check_permission(list(roles), user_roles)
                return await func(*args, **kwargs)
            return wrapper
        return decorator

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "product-service"}

@app.get("/products", response_model=List[ProductResponse])
async def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List products (public endpoint)"""
    query = db.query(Product).filter(Product.is_active == True)
    
    if category:
        query = query.filter(Product.category == category)
    
    if search:
        query = query.filter(
            Product.name.contains(search) | 
            Product.description.contains(search)
        )
    
    products = query.offset(skip).limit(limit).all()
    return products

@app.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: Session = Depends(get_db)):
    """Get product by ID (public endpoint)"""
    product = db.query(Product).filter(
        Product.id == product_id,
        Product.is_active == True
    ).first()
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return product

@app.post("/products", response_model=ProductResponse)
async def create_product(
    product_data: ProductCreate,
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """Create product (admin/manager only)"""
    PermissionChecker.check_permission(["admin", "manager"], user_roles)
    
    # Check if SKU already exists
    existing_product = db.query(Product).filter(Product.sku == product_data.sku).first()
    if existing_product:
        raise HTTPException(status_code=400, detail="Product with this SKU already exists")
    
    try:
        product = Product(**product_data.dict())
        db.add(product)
        db.commit()
        db.refresh(product)
        
        logger.info(f"Product created: {product.name} (SKU: {product.sku})")
        return product
        
    except Exception as e:
        db.rollback()
        logger.error(f"Product creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Product creation failed")

@app.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product_data: ProductUpdate,
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """Update product (admin/manager only)"""
    PermissionChecker.check_permission(["admin", "manager"], user_roles)
    
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    try:
        update_data = product_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(product, field, value)
        
        product.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(product)
        
        logger.info(f"Product updated: {product.name} (ID: {product.id})")
        return product
        
    except Exception as e:
        db.rollback()
        logger.error(f"Product update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Product update failed")

@app.delete("/products/{product_id}")
async def delete_product(
    product_id: int,
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """Delete product (admin only)"""
    PermissionChecker.check_permission(["admin"], user_roles)
    
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    try:
        # Soft delete
        product.is_active = False
        product.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Product deleted: {product.name} (ID: {product.id})")
        return {"message": "Product deleted successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Product deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Product deletion failed")

@app.put("/products/{product_id}/stock")
async def update_stock(
    product_id: int,
    quantity: int,
    operation: str = Query(..., regex="^(add|subtract|set)$"),
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """Update product stock (admin/manager only)"""
    PermissionChecker.check_permission(["admin", "manager"], user_roles)
    
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    try:
        if operation == "add":
            product.stock_quantity += quantity
        elif operation == "subtract":
            if product.stock_quantity < quantity:
                raise HTTPException(status_code=400, detail="Insufficient stock")
            product.stock_quantity -= quantity
        elif operation == "set":
            if quantity < 0:
                raise HTTPException(status_code=400, detail="Stock quantity cannot be negative")
            product.stock_quantity = quantity
        
        product.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(product)
        
        logger.info(f"Stock updated for product {product.name}: {operation} {quantity}")
        return {
            "message": "Stock updated successfully",
            "new_stock": product.stock_quantity
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Stock update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Stock update failed")

@app.get("/categories")
async def list_categories(db: Session = Depends(get_db)):
    """List all product categories"""
    categories = db.query(Product.category).filter(
        Product.is_active == True
    ).distinct().all()
    
    return [category[0] for category in categories if category[0]]

@app.get("/admin/products", response_model=List[ProductResponse])
async def list_all_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    include_inactive: bool = Query(False),
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """List all products including inactive ones (admin only)"""
    PermissionChecker.check_permission(["admin"], user_roles)
    
    query = db.query(Product)
    if not include_inactive:
        query = query.filter(Product.is_active == True)
    
    products = query.offset(skip).limit(limit).all()
    return products

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 4.2 RBAC Configuration for Keycloak
```json
# keycloak/realm-config.json
{
  "realm": "microservices",
  "enabled": true,
  "registrationAllowed": true,
  "resetPasswordAllowed": true,
  "requiredCredentials": ["password"],
  "roles": {
    "realm": [
      {
        "name": "admin",
        "description": "Administrator role with full access"
      },
      {
        "name": "manager",
        "description": "Manager role with product management access"
      },
      {
        "name": "user",
        "description": "Regular user role"
      }
    ]
  },
  "clients": [
    {
      "clientId": "api-gateway",
      "enabled": true,
      "publicClient": true,
      "directAccessGrantsEnabled": true,
      "standardFlowEnabled": true,
      "redirectUris": ["*"],
      "webOrigins": ["*"]
    }
  ],
  "users": [
    {
      "username": "admin",
      "enabled": true,
      "firstName": "System",
      "lastName": "Administrator",
      "email": "admin@example.com",
      "credentials": [
        {
          "type": "password",
          "value": "admin123",
          "temporary": false
        }
      ],
      "realmRoles": ["admin"]
    },
    {
      "username": "manager",
      "enabled": true,
      "firstName": "Product",
      "lastName": "Manager",
      "email": "manager@example.com",
      "credentials": [
        {
          "type": "password",
          "value": "manager123",
          "temporary": false
        }
      ],
      "realmRoles": ["manager"]
    }
  ]
}
```

### Phase 5: Order Service with Service-to-Service Authentication

#### 5.1 Order Service Implementation
```python
# services/order-service/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from sqlalchemy import create_engine, Column, Integer, String, DateTime, DECIMAL, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel
from typing import List, Optional
from decimal import Decimal
from datetime import datetime
import httpx
import pika
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Order Service", version="1.0.0")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres123@localhost:5432/microservices_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Order models
class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # Keycloak user ID
    status = Column(String, default="pending")  # pending, confirmed, shipped, delivered, cancelled
    total_amount = Column(DECIMAL(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    product_id = Column(Integer)
    quantity = Column(Integer)
    price = Column(DECIMAL(10, 2))
    
    order = relationship("Order", back_populates="items")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int

class OrderItemResponse(BaseModel):
    id: int
    product_id: int
    quantity: int
    price: Decimal

    class Config:
        from_attributes = True

class OrderCreate(BaseModel):
    items: List[OrderItemCreate]

class OrderResponse(BaseModel):
    id: int
    user_id: str
    status: str
    total_amount: Decimal
    created_at: datetime
    items: List[OrderItemResponse]

    class Config:
        from_attributes = True

# Dependencies
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user_id(x_user_id: str = Header(..., alias="X-User-ID")):
    """Get current user ID from gateway headers"""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID not provided")
    return x_user_id

async def get_user_roles(x_user_roles: str = Header("", alias="X-User-Roles")):
    """Get user roles from gateway headers"""
    return x_user_roles.split(",") if x_user_roles else []

class ServiceAuthenticator:
    """Handle service-to-service authentication"""
    
    def __init__(self):
        self.service_token = None
        self.token_expiry = None
    
    async def get_service_token(self):
        """Get service-to-service authentication token"""
        # In a real implementation, this would get a service account token
        # For now, we'll use the API gateway's internal token mechanism
        return "service-internal-token"
    
    async def authenticate_service_request(self, service_name: str):
        """Authenticate requests to other services"""
        token = await self.get_service_token()
        return {"Authorization": f"Bearer {token}"}

class ProductServiceClient:
    """Client for communicating with Product Service"""
    
    def __init__(self):
        self.base_url = "http://product-service:8000"
        self.auth = ServiceAuthenticator()
    
    async def get_product(self, product_id: int):
        """Get product information"""
        headers = await self.auth.authenticate_service_request("product-service")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/products/{product_id}",
                headers=headers
            )
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
            elif response.status_code != 200:
                raise HTTPException(status_code=503, detail="Product service unavailable")
            
            return response.json()
    
    async def update_stock(self, product_id: int, quantity: int, operation: str = "subtract"):
        """Update product stock"""
        headers = await self.auth.authenticate_service_request("product-service")
        
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/products/{product_id}/stock",
                params={"quantity": quantity, "operation": operation},
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to update stock for product {product_id}"
                )
            
            return response.json()

class MessageQueueService:
    """Handle message queue operations"""
    
    def __init__(self):
        self.rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://admin:admin123@rabbitmq:5672")
    
    def publish_order_event(self, event_type: str, order_data: dict):
        """Publish order events to message queue"""
        try:
            connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
            channel = connection.channel()
            
            # Declare exchange and queue
            channel.exchange_declare(exchange='orders', exchange_type='topic')
            
            # Publish message
            message = {
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": order_data
            }
            
            channel.basic_publish(
                exchange='orders',
                routing_key=f'order.{event_type}',
                body=json.dumps(message, default=str)
            )
            
            connection.close()
            logger.info(f"Published order event: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to publish order event: {str(e)}")

# Service instances
product_client = ProductServiceClient()
mq_service = MessageQueueService()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "order-service"}

@app.post("/orders", response_model=OrderResponse)
async def create_order(
    order_data: OrderCreate,
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Create a new order"""
    if not order_data.items:
        raise HTTPException(status_code=400, detail="Order must contain at least one item")
    
    try:
        # Validate products and calculate total
        order_items = []
        total_amount = Decimal('0.00')
        
        for item in order_data.items:
            # Get product info from Product Service
            product = await product_client.get_product(item.product_id)
            
            # Check stock availability
            if product["stock_quantity"] < item.quantity:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient stock for product {item.product_id}"
                )
            
            # Calculate item total
            item_price = Decimal(str(product["price"]))
            item_total = item_price * item.quantity
            total_amount += item_total
            
            order_items.append({
                "product_id": item.product_id,
                "quantity": item.quantity,
                "price": item_price
            })
        
        # Create order
        order = Order(
            user_id=current_user_id,
            total_amount=total_amount
        )
        
        db.add(order)
        db.flush()  # Get order ID
        
        # Create order items
        for item_data in order_items:
            order_item = OrderItem(
                order_id=order.id,
                **item_data
            )
            db.add(order_item)
        
        db.commit()
        db.refresh(order)
        
        # Update product stock
        for item in order_data.items:
            await product_client.update_stock(item.product_id, item.quantity, "subtract")
        
        # Publish order created event
        background_tasks.add_task(
            mq_service.publish_order_event,
            "created",
            {
                "order_id": order.id,
                "user_id": current_user_id,
                "total_amount": float(total_amount),
                "items": order_items
            }
        )
        
        logger.info(f"Order created: {order.id} for user {current_user_id}")
        return order
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Order creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Order creation failed")

@app.get("/orders", response_model=List[OrderResponse])
async def list_user_orders(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """List orders for current user"""
    orders = db.query(Order).filter(Order.user_id == current_user_id).all()
    return orders

@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: int,
    current_user_id: str = Depends(get_current_user_id),
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """Get order by ID"""
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Check ownership or admin access
    if order.user_id != current_user_id and "admin" not in user_roles:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return order

@app.put("/orders/{order_id}/status")
async def update_order_status(
    order_id: int,
    status: str,
    background_tasks: BackgroundTasks,
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """Update order status (admin only)"""
    if "admin" not in user_roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    valid_statuses = ["pending", "confirmed", "shipped", "delivered", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
    
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    try:
        old_status = order.status
        order.status = status
        order.updated_at = datetime.utcnow()
        db.commit()
        
        # Publish status update event
        background_tasks.add_task(
            mq_service.publish_order_event,
            "status_updated",
            {
                "order_id": order.id,
                "old_status": old_status,
                "new_status": status,
                "user_id": order.user_id
            }
        )
        
        logger.info(f"Order {order_id} status updated: {old_status} -> {status}")
        return {"message": "Order status updated successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Order status update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Order status update failed")

@app.get("/admin/orders", response_model=List[OrderResponse])
async def list_all_orders(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    user_roles: List[str] = Depends(get_user_roles),
    db: Session = Depends(get_db)
):
    """List all orders (admin only)"""
    if "admin" not in user_roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    query = db.query(Order)
    if status:
        query = query.filter(Order.status == status)
    
    orders = query.offset(skip).limit(limit).all()
    return orders

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Phase 6: Security Monitoring and Auditing Service

#### 6.1 Security Service Implementation
```python
# services/security-service/app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
import logging
import asyncio
from collections import defaultdict, Counter
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Security Monitoring Service", version="1.0.0")

# Elasticsearch setup
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
es_client = Elasticsearch([ELASTICSEARCH_URL])

# Security event models
class SecurityEvent(BaseModel):
    timestamp: float
    method: str
    url: str
    client_ip: str
    status_code: int
    process_time: float
    user_agent: str
    user_id: Optional[str] = None
    username: Optional[str] = None

class SecurityAlert(BaseModel):
    id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    description: str
    source_ip: str
    user_id: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any]

class AnomalyDetector:
    """Detect security anomalies and suspicious patterns"""
    
    def __init__(self):
        self.failed_login_threshold = 5
        self.request_rate_threshold = 100  # requests per minute
        self.suspicious_user_agents = [
            "sqlmap", "nmap", "nikto", "dirb", "gobuster", "burp"
        ]
        
    async def analyze_event(self, event: SecurityEvent) -> List[SecurityAlert]:
        """Analyze security event for anomalies"""
        alerts = []
        
        # Check for failed login attempts
        if "login" in event.url and event.status_code == 401:
            alerts.extend(await self._check_failed_logins(event))
        
        # Check for suspicious user agents
        if any(agent in event.user_agent.lower() for agent in self.suspicious_user_agents):
            alerts.append(await self._create_suspicious_agent_alert(event))
        
        # Check for high request rates
        alerts.extend(await self._check_request_rate(event))
        
        # Check for unusual response times
        if event.process_time > 10.0:  # Very slow response
            alerts.append(await self._create_slow_response_alert(event))
        
        # Check for SQL injection patterns
        if self._detect_sql_injection(event.url):
            alerts.append(await self._create_sql_injection_alert(event))
        
        return alerts
    
    async def _check_failed_logins(self, event: SecurityEvent) -> List[SecurityAlert]:
        """Check for repeated failed login attempts"""
        alerts = []
        
        # Query recent failed logins from this IP
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"client_ip": event.client_ip}},
                        {"match": {"status_code": 401}},
                        {"range": {"timestamp": {"gte": "now-5m"}}}
                    ]
                }
            }
        }
        
        try:
            result = es_client.search(index="security-events", body=query)
            failed_count = result["hits"]["total"]["value"]
            
            if failed_count >= self.failed_login_threshold:
                alert = SecurityAlert(
                    id=self._generate_alert_id(event),
                    alert_type="brute_force_attack",
                    severity="high",
                    description=f"Potential brute force attack: {failed_count} failed login attempts from {event.client_ip}",
                    source_ip=event.client_ip,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "failed_attempts": failed_count,
                        "time_window": "5 minutes"
                    }
                )
                alerts.append(alert)
                
        except Exception as e:
            logger.error(f"Failed to check failed logins: {str(e)}")
        
        return alerts
    
    async def _create_suspicious_agent_alert(self, event: SecurityEvent) -> SecurityAlert:
        """Create alert for suspicious user agent"""
        return SecurityAlert(
            id=self._generate_alert_id(event),
            alert_type="suspicious_user_agent",
            severity="medium",
            description=f"Suspicious user agent detected: {event.user_agent}",
            source_ip=event.client_ip,
            user_id=event.user_id,
            timestamp=datetime.utcnow(),
            metadata={
                "user_agent": event.user_agent,
                "url": event.url
            }
        )
    
    async def _check_request_rate(self, event: SecurityEvent) -> List[SecurityAlert]:
        """Check for high request rates"""
        alerts = []
        
        # Query requests from this IP in the last minute
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"client_ip": event.client_ip}},
                        {"range": {"timestamp": {"gte": "now-1m"}}}
                    ]
                }
            }
        }
        
        try:
            result = es_client.search(index="security-events", body=query)
            request_count = result["hits"]["total"]["value"]
            
            if request_count > self.request_rate_threshold:
                alert = SecurityAlert(
                    id=self._generate_alert_id(event),
                    alert_type="high_request_rate",
                    severity="medium",
                    description=f"High request rate detected: {request_count} requests/minute from {event.client_ip}",
                    source_ip=event.client_ip,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "requests_per_minute": request_count,
                        "threshold": self.request_rate_threshold
                    }
                )
                alerts.append(alert)
                
        except Exception as e:
            logger.error(f"Failed to check request rate: {str(e)}")
        
        return alerts
    
    async def _create_slow_response_alert(self, event: SecurityEvent) -> SecurityAlert:
        """Create alert for slow response times"""
        return SecurityAlert(
            id=self._generate_alert_id(event),
            alert_type="slow_response",
            severity="low",
            description=f"Slow response detected: {event.process_time:.2f}s for {event.url}",
            source_ip=event.client_ip,
            user_id=event.user_id,
            timestamp=datetime.utcnow(),
            metadata={
                "response_time": event.process_time,
                "url": event.url,
                "method": event.method
            }
        )
    
    def _detect_sql_injection(self, url: str) -> bool:
        """Simple SQL injection pattern detection"""
        sql_patterns = [
            "union select", "or 1=1", "'; drop table", "' or '1'='1",
            "exec(", "script>", "<iframe", "javascript:"
        ]
        
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in sql_patterns)
    
    async def _create_sql_injection_alert(self, event: SecurityEvent) -> SecurityAlert:
        """Create alert for potential SQL injection"""
        return SecurityAlert(
            id=self._generate_alert_id(event),
            alert_type="sql_injection_attempt",
            severity="critical",
            description=f"Potential SQL injection attempt detected in URL: {event.url}",
            source_ip=event.client_ip,
            user_id=event.user_id,
            timestamp=datetime.utcnow(),
            metadata={
                "url": event.url,
                "method": event.method,
                "user_agent": event.user_agent
            }
        )
    
    def _generate_alert_id(self, event: SecurityEvent) -> str:
        """Generate unique alert ID"""
        data = f"{event.timestamp}{event.client_ip}{event.url}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

class AuditLogger:
    """Handle audit logging"""
    
    def __init__(self):
        self.audit_index = "audit-logs"
    
    async def log_audit_event(self, event_type: str, user_id: str, resource: str, 
                            action: str, result: str, metadata: Dict[str, Any] = None):
        """Log audit event"""
        audit_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "result": result,
            "metadata": metadata or {}
        }
        
        try:
            es_client.index(index=self.audit_index, body=audit_event)
            logger.info(f"Audit event logged: {event_type} - {action} on {resource}")
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")

# Service instances
anomaly_detector = AnomalyDetector()
audit_logger = AuditLogger()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "security-service"}

@app.post("/events")
async def receive_security_event(
    event: SecurityEvent,
    background_tasks: BackgroundTasks
):
    """Receive and process security events"""
    try:
        # Store event in Elasticsearch
        event_doc = event.dict()
        event_doc["@timestamp"] = datetime.fromtimestamp(event.timestamp).isoformat()
        
        es_client.index(index="security-events", body=event_doc)
        
        # Analyze event for anomalies in background
        background_tasks.add_task(analyze_and_alert, event)
        
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"Failed to process security event: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process event")

async def analyze_and_alert(event: SecurityEvent):
    """Analyze event and create alerts"""
    try:
        alerts = await anomaly_detector.analyze_event(event)
        
        for alert in alerts:
            # Store alert
            alert_doc = alert.dict()
            alert_doc["@timestamp"] = alert.timestamp.isoformat()
            
            es_client.index(index="security-alerts", body=alert_doc)
            
            # Send critical alerts to notification system
            if alert.severity == "critical":
                await send_critical_alert_notification(alert)
                
    except Exception as e:
        logger.error(f"Failed to analyze event: {str(e)}")

async def send_critical_alert_notification(alert: SecurityAlert):
    """Send critical alert notifications"""
    # In a real implementation, this would send notifications via:
    # - Email
    # - Slack/Teams
    # - PagerDuty
    # - SMS
    logger.critical(f"CRITICAL SECURITY ALERT: {alert.description}")

@app.get("/alerts", response_model=List[SecurityAlert])
async def get_security_alerts(
    severity: Optional[str] = None,
    limit: int = 100,
    hours: int = 24
):
    """Get security alerts"""
    query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": f"now-{hours}h"}}}
                ]
            }
        },
        "sort": [{"@timestamp": {"order": "desc"}}],
        "size": limit
    }
    
    if severity:
        query["query"]["bool"]["must"].append({"match": {"severity": severity}})
    
    try:
        result = es_client.search(index="security-alerts", body=query)
        alerts = []
        
        for hit in result["hits"]["hits"]:
            alert_data = hit["_source"]
            alert = SecurityAlert(**alert_data)
            alerts.append(alert)
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get security dashboard statistics"""
    try:
        # Get stats for the last 24 hours
        stats = {}
        
        # Total events
        total_events_query = {
            "query": {"range": {"@timestamp": {"gte": "now-24h"}}}
        }
        total_events = es_client.count(index="security-events", body=total_events_query)
        stats["total_events"] = total_events["count"]
        
        # Total alerts by severity
        alerts_agg_query = {
            "query": {"range": {"@timestamp": {"gte": "now-24h"}}},
            "aggs": {
                "by_severity": {
                    "terms": {"field": "severity.keyword"}
                }
            }
        }
        alerts_result = es_client.search(index="security-alerts", body=alerts_agg_query)
        stats["alerts_by_severity"] = {
            bucket["key"]: bucket["doc_count"]
            for bucket in alerts_result["aggregations"]["by_severity"]["buckets"]
        }
        
        # Top source IPs
        top_ips_query = {
            "query": {"range": {"@timestamp": {"gte": "now-24h"}}},
            "aggs": {
                "top_ips": {
                    "terms": {"field": "client_ip.keyword", "size": 10}
                }
            }
        }
        ips_result = es_client.search(index="security-events", body=top_ips_query)
        stats["top_source_ips"] = [
            {"ip": bucket["key"], "requests": bucket["doc_count"]}
            for bucket in ips_result["aggregations"]["top_ips"]["buckets"]
        ]
        
        # Failed login attempts
        failed_logins_query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": "now-24h"}}},
                        {"match": {"status_code": 401}},
                        {"wildcard": {"url": "*login*"}}
                    ]
                }
            }
        }
        failed_logins = es_client.count(index="security-events", body=failed_logins_query)
        stats["failed_logins"] = failed_logins["count"]
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

@app.post("/audit")
async def create_audit_log(
    event_type: str,
    user_id: str,
    resource: str,
    action: str,
    result: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Create audit log entry"""
    try:
        await audit_logger.log_audit_event(
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            metadata=metadata
        )
        
        return {"status": "logged"}
        
    except Exception as e:
        logger.error(f"Failed to create audit log: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create audit log")

@app.get("/audit")
async def get_audit_logs(
    user_id: Optional[str] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
    hours: int = 24,
    limit: int = 100
):
    """Get audit logs"""
    query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"timestamp": {"gte": f"now-{hours}h"}}}
                ]
            }
        },
        "sort": [{"timestamp": {"order": "desc"}}],
        "size": limit
    }
    
    # Add filters
    if user_id:
        query["query"]["bool"]["must"].append({"match": {"user_id": user_id}})
    if resource:
        query["query"]["bool"]["must"].append({"match": {"resource": resource}})
    if action:
        query["query"]["bool"]["must"].append({"match": {"action": action}})
    
    try:
        result = es_client.search(index="audit-logs", body=query)
        logs = [hit["_source"] for hit in result["hits"]["hits"]]
        return logs
        
    except Exception as e:
        logger.error(f"Failed to get audit logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit logs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 6.2 Security Monitoring and Auditing Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'

  - job_name: 'user-service'
    static_configs:
      - targets: ['user-service:8000']
    metrics_path: '/metrics'

  - job_name: 'product-service'
    static_configs:
      - targets: ['product-service:8000']
    metrics_path: '/metrics'

  - job_name: 'order-service'
    static_configs:
      - targets: ['order-service:8000']
    metrics_path: '/metrics'

  - job_name: 'security-service'
    static_configs:
      - targets: ['security-service:8000']
    metrics_path: '/metrics'
```

```yaml
# monitoring/alert_rules.yml
groups:
- name: security_alerts
  rules:
  - alert: HighFailedLoginRate
    expr: rate(failed_logins_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High failed login rate detected"
      description: "Failed login rate is {{ $value }} per second from {{ $labels.source_ip }}"

  - alert: CriticalSecurityEvent
    expr: increase(security_events_total{severity="critical"}[5m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Critical security event detected"
      description: "Critical security event: {{ $labels.event_type }}"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "{{ $labels.instance }} has been down for more than 1 minute"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time"
      description: "95th percentile response time is {{ $value }}s"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate"
      description: "Error rate is {{ $value | humanizePercentage }}"

- name: resource_alerts
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage_percent > 80
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: HighMemoryUsage
    expr: memory_usage_percent > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: DiskSpaceLow
    expr: disk_free_percent < 10
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Low disk space"
      description: "Disk space is {{ $value }}% free on {{ $labels.instance }}"
```

### Phase 7: Deployment and Operations

### Phase 8: Monitoring and Maintenance

### Learning Objectives and Assessment

#### Project Learning Outcomes

By completing this project, you will have:

**✅ Technical Skills:**
- Built a production-ready microservices architecture
- Implemented OAuth2/OpenID Connect authentication
- Created service-to-service authentication mechanisms
- Developed comprehensive security monitoring and auditing
- Configured role-based access control (RBAC)
- Set up monitoring, alerting, and observability
- Implemented API rate limiting and security headers
- Created automated testing suites for security validation

**✅ Security Knowledge:**
- Understanding of modern authentication and authorization patterns
- Knowledge of common security vulnerabilities and mitigations
- Experience with security monitoring and incident response
- Familiarity with compliance and audit requirements
- Understanding of defense-in-depth strategies

**✅ DevOps and Operations:**
- Docker containerization and orchestration
- Kubernetes deployment strategies
- Monitoring and alerting setup
- CI/CD pipeline implementation
- Infrastructure as Code practices

#### Self-Assessment Checklist

Before considering this project complete, ensure you can:

□ **Authentication & Authorization**
  - Explain OAuth2 and OpenID Connect flows
  - Implement JWT token validation and refresh
  - Configure Keycloak for identity management
  - Set up role-based access control

□ **Security Implementation**
  - Implement API rate limiting
  - Add security headers and CORS configuration
  - Create input validation and sanitization
  - Implement audit logging and monitoring

□ **Microservices Architecture**
  - Design service boundaries and communication patterns
  - Implement service discovery and load balancing
  - Handle inter-service authentication
  - Implement circuit breakers and retry mechanisms

□ **Monitoring & Observability**
  - Set up metrics collection and visualization
  - Configure alerting for security events
  - Implement log aggregation and analysis
  - Create dashboards for operational visibility

□ **Testing & Validation**
  - Write security-focused unit and integration tests
  - Perform load testing and security scanning
  - Validate RBAC and access controls
  - Test incident response procedures

#### Extended Challenges

To further enhance your skills, consider these additional challenges:

1. **Multi-Tenant Architecture**: Extend the system to support multiple tenants with data isolation

2. **Advanced Security Features**:
   - Implement OAuth2 PKCE for public clients
   - Add multi-factor authentication (MFA)
   - Implement device fingerprinting
   - Add behavioral analysis for anomaly detection

3. **Compliance Integration**:
   - Add GDPR compliance features (data portability, right to deletion)
   - Implement SOX audit trails
   - Add PCI DSS compliance for payment data

4. **Advanced Monitoring**:
   - Implement distributed tracing with Jaeger
   - Add application performance monitoring (APM)
   - Create custom security dashboards
   - Implement automated incident response

5. **High Availability & Disaster Recovery**:
   - Implement database replication and failover
   - Add cross-region deployment
   - Create backup and restore procedures
   - Implement chaos engineering tests

### Resources for Further Learning

#### Books
- "Microservices Security in Action" by Prabath Siriwardena
- "OAuth 2 in Action" by Justin Richer and Antonio Sanso
- "Building Microservices" by Sam Newman
- "Security Engineering" by Ross Anderson

#### Online Courses
- OWASP Web Security Testing Guide
- Keycloak Administration and Development
- Kubernetes Security Best Practices
- Microservices Architecture Patterns

#### Tools and Platforms
- **Security Testing**: OWASP ZAP, Burp Suite, Nessus
- **Static Analysis**: SonarQube, Checkmarx, Veracode
- **Container Security**: Aqua Security, Twistlock, Falco
- **Cloud Security**: AWS Security Hub, Azure Security Center

#### Community and Standards
- OWASP (Open Web Application Security Project)
- NIST Cybersecurity Framework
- ISO 27001 Security Standards
- Cloud Security Alliance (CSA)

---

**Congratulations!** 🎉 You have successfully built a comprehensive secure microservices architecture with modern authentication, monitoring, and security features. This project demonstrates production-ready skills in microservices development, security implementation, and DevOps practices.
