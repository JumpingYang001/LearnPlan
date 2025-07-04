# Security in Microservices

*Duration: 2-3 weeks*

## Overview

Security in microservices architecture presents unique challenges compared to monolithic applications. With multiple services communicating over networks, each service boundary becomes a potential attack vector. This comprehensive guide covers authentication, authorization, API security, secure communication, and implementation of robust security patterns.

## Key Security Challenges in Microservices

### 1. Distributed Attack Surface
- Multiple entry points across services
- Network communication vulnerabilities
- Service-to-service trust relationships
- Data consistency across security boundaries

### 2. Identity and Access Management Complexity
- User identity propagation across services
- Service identity management
- Token lifecycle management
- Role-based access control (RBAC) distribution

### 3. Network Security
- Inter-service communication encryption
- API gateway security
- Service mesh security policies
- Network segmentation and isolation

## Core Security Concepts

### Authentication vs Authorization

#### Authentication (AuthN) - "Who are you?"
Authentication verifies the identity of users or services attempting to access your system.

```python
# JWT-based Authentication Example
import jwt
import datetime
from functools import wraps
from flask import Flask, request, jsonify

app = Flask(__name__)
SECRET_KEY = 'your-secret-key-change-in-production'

class AuthenticationService:
    @staticmethod
    def generate_token(user_id, username, roles=None):
        """Generate JWT token with user information"""
        payload = {
            'user_id': user_id,
            'username': username,
            'roles': roles or [],
            'iat': datetime.datetime.utcnow(),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        return token
    
    @staticmethod
    def verify_token(token):
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")

# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            payload = AuthenticationService.verify_token(token)
            request.current_user = payload
            
        except Exception as e:
            return jsonify({'error': str(e)}), 401
        
        return f(*args, **kwargs)
    
    return decorated

@app.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # In production, verify against database with hashed passwords
    if username == 'admin' and password == 'secret':
        token = AuthenticationService.generate_token(
            user_id=1, 
            username=username, 
            roles=['admin', 'user']
        )
        return jsonify({'token': token})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/protected', methods=['GET'])
@require_auth
def protected_route():
    """Protected endpoint requiring authentication"""
    return jsonify({
        'message': 'Access granted!',
        'user': request.current_user
    })
```

#### Authorization (AuthZ) - "What can you do?"
Authorization determines what actions an authenticated user or service can perform.

```python
# Role-Based Access Control (RBAC) Example
from functools import wraps
from flask import jsonify, request

class AuthorizationService:
    # Define role hierarchy and permissions
    PERMISSIONS = {
        'admin': ['read', 'write', 'delete', 'manage_users'],
        'manager': ['read', 'write', 'delete'],
        'user': ['read', 'write'],
        'guest': ['read']
    }
    
    @classmethod
    def has_permission(cls, user_roles, required_permission):
        """Check if user roles include required permission"""
        for role in user_roles:
            if required_permission in cls.PERMISSIONS.get(role, []):
                return True
        return False

def require_permission(permission):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'error': 'Authentication required'}), 401
            
            user_roles = request.current_user.get('roles', [])
            
            if not AuthorizationService.has_permission(user_roles, permission):
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated
    return decorator

@app.route('/admin/users', methods=['GET'])
@require_auth
@require_permission('manage_users')
def manage_users():
    """Admin-only endpoint for user management"""
    return jsonify({'users': ['user1', 'user2', 'user3']})

@app.route('/data', methods=['DELETE'])
@require_auth
@require_permission('delete')
def delete_data():
    """Endpoint requiring delete permission"""
    return jsonify({'message': 'Data deleted successfully'})

@app.route('/data', methods=['GET'])
@require_auth
@require_permission('read')
def read_data():
    """Endpoint requiring read permission"""
    return jsonify({'data': 'sensitive information'})
```

## API Security Best Practices

### 1. Input Validation and Sanitization

```python
from marshmallow import Schema, fields, ValidationError
from flask import request, jsonify

class UserSchema(Schema):
    """Input validation schema"""
    username = fields.Str(required=True, validate=fields.Length(min=3, max=50))
    email = fields.Email(required=True)
    age = fields.Int(validate=fields.Range(min=18, max=120))
    role = fields.Str(validate=fields.OneOf(['user', 'admin', 'manager']))

def validate_input(schema_class):
    """Decorator for input validation"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            schema = schema_class()
            try:
                validated_data = schema.load(request.get_json())
                request.validated_data = validated_data
            except ValidationError as err:
                return jsonify({'errors': err.messages}), 400
            
            return f(*args, **kwargs)
        return decorated
    return decorator

@app.route('/users', methods=['POST'])
@require_auth
@validate_input(UserSchema)
@require_permission('write')
def create_user():
    """Create user with input validation"""
    user_data = request.validated_data
    
    # Additional business logic validation
    if user_data['role'] == 'admin' and 'admin' not in request.current_user['roles']:
        return jsonify({'error': 'Cannot create admin user'}), 403
    
    # Create user logic here
    return jsonify({'message': 'User created', 'user': user_data}), 201
```

### 2. Rate Limiting and Throttling

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis

# Initialize rate limiter with Redis backend
limiter = Limiter(
    app,
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)

class AdvancedRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def check_rate_limit(self, key, limit, window):
        """Sliding window rate limiting"""
        current_time = int(time.time())
        pipeline = self.redis.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, current_time - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(current_time): current_time})
        
        # Set expiry
        pipeline.expire(key, window)
        
        results = pipeline.execute()
        current_requests = results[1]
        
        return current_requests < limit

# Apply rate limiting to endpoints
@app.route('/api/sensitive', methods=['POST'])
@limiter.limit("5 per minute")  # Global rate limit
@require_auth
def sensitive_endpoint():
    # User-specific rate limiting
    user_id = request.current_user['user_id']
    rate_limiter = AdvancedRateLimiter(redis.Redis())
    
    if not rate_limiter.check_rate_limit(f"user:{user_id}", limit=10, window=3600):
        return jsonify({'error': 'User rate limit exceeded'}), 429
    
    return jsonify({'message': 'Success'})
```

### 3. API Versioning and Security Headers

```python
from flask import Blueprint

# API versioning
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')
api_v2 = Blueprint('api_v2', __name__, url_prefix='/api/v2')

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Remove server information
    response.headers.pop('Server', None)
    
    return response

@api_v1.route('/users')
def get_users_v1():
    """Version 1 API - deprecated"""
    return jsonify({
        'version': '1.0',
        'deprecated': True,
        'migration_guide': '/api/v2/docs'
    })

@api_v2.route('/users')
@require_auth
def get_users_v2():
    """Version 2 API - current"""
    return jsonify({
        'version': '2.0',
        'users': []
    })
```

## Service-to-Service Communication Security

### 1. Mutual TLS (mTLS) Authentication

```python
import ssl
import requests
from cryptography import x509
from cryptography.x509.oid import NameOID

class MTLSClient:
    def __init__(self, cert_file, key_file, ca_file):
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file
    
    def create_session(self):
        """Create requests session with mTLS configuration"""
        session = requests.Session()
        session.cert = (self.cert_file, self.key_file)
        session.verify = self.ca_file
        
        return session
    
    def call_service(self, url, data=None):
        """Make authenticated service call"""
        session = self.create_session()
        
        try:
            if data:
                response = session.post(url, json=data)
            else:
                response = session.get(url)
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.SSLError as e:
            raise Exception(f"SSL/TLS error: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error: {e}")

# Service certificate validation
class CertificateValidator:
    def __init__(self, allowed_services):
        self.allowed_services = allowed_services
    
    def validate_client_cert(self, cert_pem):
        """Validate client certificate"""
        try:
            cert = x509.load_pem_x509_certificate(cert_pem.encode())
            
            # Check certificate validity
            now = datetime.datetime.utcnow()
            if cert.not_valid_after < now or cert.not_valid_before > now:
                return False, "Certificate expired or not yet valid"
            
            # Extract service name from certificate
            subject = cert.subject
            service_name = None
            
            for attribute in subject:
                if attribute.oid == NameOID.COMMON_NAME:
                    service_name = attribute.value
                    break
            
            if service_name not in self.allowed_services:
                return False, f"Service {service_name} not authorized"
            
            return True, service_name
        
        except Exception as e:
            return False, f"Certificate validation error: {e}"

# Flask middleware for mTLS
@app.before_request
def verify_client_certificate():
    """Verify client certificate for service-to-service calls"""
    if request.path.startswith('/internal/'):
        cert_header = request.headers.get('X-Client-Cert')
        
        if not cert_header:
            return jsonify({'error': 'Client certificate required'}), 401
        
        validator = CertificateValidator(['user-service', 'order-service', 'payment-service'])
        is_valid, result = validator.validate_client_cert(cert_header)
        
        if not is_valid:
            return jsonify({'error': result}), 403
        
        request.client_service = result
```

### 2. Service Mesh Security (Istio Example)

```yaml
# Istio Service Mesh Security Configuration
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: STRICT  # Require mTLS for all communication

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: service-authorization
  namespace: production
spec:
  selector:
    matchLabels:
      app: user-service
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/production/sa/order-service"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/users/*"]
  - from:
    - source:
        principals: ["cluster.local/ns/production/sa/payment-service"]
    to:
    - operation:
        methods: ["GET"]
        paths: ["/api/users/*/profile"]
```

```python
# Service mesh integration in Python
import os
from flask import Flask, request

class ServiceMeshIntegration:
    def __init__(self):
        self.namespace = os.getenv('KUBERNETES_NAMESPACE', 'default')
        self.service_account = os.getenv('KUBERNETES_SERVICE_ACCOUNT', 'default')
    
    def get_caller_identity(self):
        """Extract caller identity from Istio headers"""
        # Istio automatically injects these headers
        source_service = request.headers.get('X-Forwarded-Client-Cert')
        source_namespace = request.headers.get('X-Source-Namespace')
        source_sa = request.headers.get('X-Source-Service-Account')
        
        return {
            'service': source_service,
            'namespace': source_namespace,
            'service_account': source_sa
        }
    
    def verify_caller(self, allowed_services):
        """Verify if caller is authorized"""
        caller = self.get_caller_identity()
        
        if not caller['service']:
            return False, "No caller identity found"
        
        if caller['service'] not in allowed_services:
            return False, f"Service {caller['service']} not authorized"
        
        return True, caller

# Usage in Flask app
service_mesh = ServiceMeshIntegration()

@app.route('/internal/user-data')
def get_user_data():
    """Internal endpoint for service-to-service communication"""
    is_authorized, result = service_mesh.verify_caller(['order-service', 'payment-service'])
    
    if not is_authorized:
        return jsonify({'error': result}), 403
    
    return jsonify({'data': 'sensitive user data', 'caller': result})
```

## OAuth 2.0 and OpenID Connect Implementation

### 1. OAuth 2.0 Authorization Server

```python
from authlib.integrations.flask_oauth2 import AuthorizationServer, ResourceProtector
from authlib.oauth2.rfc6749 import grants
from authlib.common.security import generate_token

class OAuth2Server:
    def __init__(self, app):
        self.app = app
        self.server = AuthorizationServer(app)
        self.setup_oauth2()
    
    def setup_oauth2(self):
        """Configure OAuth2 server"""
        # Register supported grants
        self.server.register_grant(grants.AuthorizationCodeGrant)
        self.server.register_grant(grants.ClientCredentialsGrant)
        self.server.register_grant(grants.RefreshTokenGrant)
        
        # Configure token endpoints
        self.server.register_endpoint('RevocationEndpoint')

class ClientCredentialsGrant(grants.ClientCredentialsGrant):
    """Client Credentials Grant for service-to-service auth"""
    
    def save_token(self, token, request, *args, **kwargs):
        """Save issued token"""
        # In production, save to database
        client_id = request.client.client_id
        access_token = token['access_token']
        expires_in = token['expires_in']
        
        # Store token with client information
        token_store[access_token] = {
            'client_id': client_id,
            'scope': token.get('scope', ''),
            'expires_at': time.time() + expires_in
        }

# OAuth2 protected resource decorator
require_oauth = ResourceProtector()

@app.route('/oauth/token', methods=['POST'])
def issue_token():
    """OAuth2 token endpoint"""
    return oauth_server.create_token_response()

@app.route('/api/protected', methods=['GET'])
@require_oauth('read')
def protected_resource():
    """OAuth2 protected resource"""
    token = request.oauth_token
    return jsonify({
        'message': 'Access granted',
        'client_id': token['client_id'],
        'scope': token['scope']
    })
```

### 2. OpenID Connect Integration

```python
from authlib.integrations.flask_client import OAuth
from authlib.oidc.core import CodeIDToken

oauth = OAuth(app)

# Configure OpenID Connect provider
google = oauth.register(
    name='google',
    client_id='your-google-client-id',
    client_secret='your-google-client-secret',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

class OpenIDConnectHandler:
    def __init__(self, oauth_client):
        self.oauth_client = oauth_client
    
    def initiate_login(self, redirect_uri):
        """Initiate OpenID Connect login flow"""
        return self.oauth_client.authorize_redirect(redirect_uri)
    
    def handle_callback(self, request):
        """Handle OpenID Connect callback"""
        token = self.oauth_client.authorize_access_token()
        
        # Verify ID token
        id_token = token.get('id_token')
        if id_token:
            claims = self.oauth_client.parse_id_token(token)
            
            user_info = {
                'sub': claims['sub'],
                'email': claims.get('email'),
                'name': claims.get('name'),
                'picture': claims.get('picture')
            }
            
            return user_info
        
        return None

@app.route('/auth/login')
def login():
    """Initiate OIDC login"""
    oidc_handler = OpenIDConnectHandler(google)
    redirect_uri = url_for('auth_callback', _external=True)
    return oidc_handler.initiate_login(redirect_uri)

@app.route('/auth/callback')
def auth_callback():
    """Handle OIDC callback"""
    oidc_handler = OpenIDConnectHandler(google)
    user_info = oidc_handler.handle_callback(request)
    
    if user_info:
        # Create internal session/token
        token = AuthenticationService.generate_token(
            user_id=user_info['sub'],
            username=user_info['email'],
            roles=['user']
        )
        
        return jsonify({'token': token, 'user': user_info})
    
    return jsonify({'error': 'Authentication failed'}), 401
```

## API Gateway Security

### 1. Gateway as Security Perimeter

```python
# API Gateway Security Implementation
from flask import Flask, request, jsonify
import requests
import time
import hashlib
import hmac

class SecureAPIGateway:
    def __init__(self):
        self.services = {
            'user-service': 'http://user-service:8080',
            'order-service': 'http://order-service:8080',
            'payment-service': 'http://payment-service:8080'
        }
        self.api_keys = {
            'client1': {
                'key': 'sk_live_abc123',
                'secret': 'sk_secret_xyz789',
                'rate_limit': 1000,
                'permissions': ['read', 'write']
            }
        }
    
    def authenticate_api_key(self, api_key, signature, timestamp, body):
        """Authenticate API key with HMAC signature"""
        client_config = self.api_keys.get(api_key)
        if not client_config:
            return False, "Invalid API key"
        
        # Check timestamp to prevent replay attacks
        current_time = int(time.time())
        if abs(current_time - int(timestamp)) > 300:  # 5 minutes tolerance
            return False, "Request timestamp too old"
        
        # Verify HMAC signature
        expected_signature = hmac.new(
            client_config['secret'].encode(),
            f"{timestamp}.{body}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            return False, "Invalid signature"
        
        return True, client_config
    
    def route_request(self, service_name, path, method, headers, data):
        """Route request to appropriate microservice"""
        if service_name not in self.services:
            return {'error': 'Service not found'}, 404
        
        service_url = f"{self.services[service_name]}{path}"
        
        # Add internal authentication headers
        internal_headers = headers.copy()
        internal_headers['X-Gateway-Auth'] = 'internal-token'
        internal_headers['X-Request-ID'] = self.generate_request_id()
        
        try:
            response = requests.request(
                method=method,
                url=service_url,
                headers=internal_headers,
                json=data,
                timeout=30
            )
            
            return response.json(), response.status_code
        
        except requests.exceptions.Timeout:
            return {'error': 'Service timeout'}, 504
        except requests.exceptions.ConnectionError:
            return {'error': 'Service unavailable'}, 503

# Gateway middleware
app = Flask(__name__)
gateway = SecureAPIGateway()

@app.before_request
def gateway_security():
    """Apply gateway-level security"""
    # Skip internal health checks
    if request.path in ['/health', '/metrics']:
        return
    
    # Validate API key and signature
    api_key = request.headers.get('X-API-Key')
    signature = request.headers.get('X-Signature')
    timestamp = request.headers.get('X-Timestamp')
    
    if not all([api_key, signature, timestamp]):
        return jsonify({'error': 'Missing authentication headers'}), 401
    
    body = request.get_data(as_text=True)
    is_valid, result = gateway.authenticate_api_key(api_key, signature, timestamp, body)
    
    if not is_valid:
        return jsonify({'error': result}), 401
    
    request.client_config = result

@app.route('/<service>/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_request(service, path):
    """Proxy requests to microservices"""
    data = request.get_json() if request.is_json else None
    
    result, status_code = gateway.route_request(
        service_name=service,
        path=f"/{path}",
        method=request.method,
        headers=dict(request.headers),
        data=data
    )
    
    return jsonify(result), status_code
```

### 2. Request/Response Transformation and Validation

```python
from marshmallow import Schema, fields, ValidationError
import json

class APIGatewayValidator:
    def __init__(self):
        self.schemas = {
            'user-service': {
                'POST:/users': self.UserCreateSchema,
                'PUT:/users/{id}': self.UserUpdateSchema
            }
        }
    
    class UserCreateSchema(Schema):
        username = fields.Str(required=True, validate=fields.Length(min=3, max=50))
        email = fields.Email(required=True)
        password = fields.Str(required=True, validate=fields.Length(min=8))
        age = fields.Int(validate=fields.Range(min=18))
    
    class UserUpdateSchema(Schema):
        username = fields.Str(validate=fields.Length(min=3, max=50))
        email = fields.Email()
        age = fields.Int(validate=fields.Range(min=18))
    
    def validate_request(self, service, endpoint, data):
        """Validate request against schema"""
        schema_key = f"{endpoint}"
        schema_class = self.schemas.get(service, {}).get(schema_key)
        
        if not schema_class:
            return True, data  # No validation defined
        
        try:
            schema = schema_class()
            validated_data = schema.load(data)
            return True, validated_data
        except ValidationError as err:
            return False, err.messages

class ResponseTransformer:
    """Transform responses for different API versions"""
    
    def transform_user_response_v1_to_v2(self, v1_response):
        """Transform user response from v1 to v2 format"""
        if 'user' in v1_response:
            user = v1_response['user']
            return {
                'data': {
                    'id': user.get('id'),
                    'profile': {
                        'username': user.get('username'),
                        'email': user.get('email')
                    },
                    'metadata': {
                        'created_at': user.get('created_at'),
                        'last_login': user.get('last_login')
                    }
                },
                'version': '2.0'
            }
        return v1_response
    
    def apply_field_filtering(self, response, allowed_fields):
        """Filter response fields based on client permissions"""
        if isinstance(response, dict):
            return {k: v for k, v in response.items() if k in allowed_fields}
        return response

# Enhanced gateway with validation and transformation
@app.route('/<service>/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def enhanced_proxy_request(service, path):
    """Enhanced proxy with validation and transformation"""
    validator = APIGatewayValidator()
    transformer = ResponseTransformer()
    
    # Request validation
    if request.is_json:
        data = request.get_json()
        endpoint = f"{request.method}:/{path}"
        
        is_valid, result = validator.validate_request(service, endpoint, data)
        if not is_valid:
            return jsonify({'errors': result}), 400
        
        data = result
    else:
        data = None
    
    # Route to service
    result, status_code = gateway.route_request(
        service_name=service,
        path=f"/{path}",
        method=request.method,
        headers=dict(request.headers),
        data=data
    )
    
    # Response transformation
    client_version = request.headers.get('API-Version', '1.0')
    if client_version == '2.0' and service == 'user-service':
        result = transformer.transform_user_response_v1_to_v2(result)
    
    # Field filtering based on permissions
    client_permissions = request.client_config.get('permissions', [])
    if 'read_sensitive' not in client_permissions:
        allowed_fields = ['id', 'username', 'email']  # Exclude sensitive fields
        result = transformer.apply_field_filtering(result, allowed_fields)
    
    return jsonify(result), status_code
```

## Security Patterns and Best Practices

### 1. Zero Trust Architecture

```python
class ZeroTrustValidator:
    """Implement Zero Trust principles"""
    
    def __init__(self):
        self.trusted_networks = []  # Empty - trust no network
        self.device_registry = {}
        self.risk_engine = RiskAssessmentEngine()
    
    def validate_request(self, request_context):
        """Validate every request regardless of source"""
        validation_results = {
            'device_trust': self.validate_device(request_context.device_id),
            'user_behavior': self.analyze_user_behavior(request_context.user_id),
            'network_risk': self.assess_network_risk(request_context.source_ip),
            'resource_sensitivity': self.classify_resource(request_context.resource)
        }
        
        risk_score = self.risk_engine.calculate_risk(validation_results)
        
        if risk_score > 0.8:
            return False, "High risk - access denied"
        elif risk_score > 0.5:
            return True, "Medium risk - additional verification required"
        else:
            return True, "Low risk - access granted"
    
    def validate_device(self, device_id):
        """Validate device trust level"""
        device = self.device_registry.get(device_id)
        if not device:
            return {'trust_level': 0, 'reason': 'Unknown device'}
        
        # Check device health
        health_checks = {
            'os_updated': device.get('os_version') in ['latest', 'supported'],
            'antivirus_active': device.get('antivirus_status') == 'active',
            'encryption_enabled': device.get('disk_encryption') == True,
            'compliance_met': device.get('compliance_score', 0) > 0.7
        }
        
        trust_level = sum(health_checks.values()) / len(health_checks)
        return {'trust_level': trust_level, 'checks': health_checks}

class RiskAssessmentEngine:
    """Assess risk for access decisions"""
    
    def calculate_risk(self, validation_results):
        """Calculate overall risk score"""
        weights = {
            'device_trust': 0.3,
            'user_behavior': 0.3,
            'network_risk': 0.2,
            'resource_sensitivity': 0.2
        }
        
        risk_score = 0
        for factor, weight in weights.items():
            factor_risk = self.normalize_risk(validation_results.get(factor, {}))
            risk_score += factor_risk * weight
        
        return min(1.0, risk_score)
    
    def normalize_risk(self, factor_data):
        """Convert factor data to risk score (0-1)"""
        if isinstance(factor_data, dict):
            if 'trust_level' in factor_data:
                return 1 - factor_data['trust_level']
            elif 'risk_score' in factor_data:
                return factor_data['risk_score']
        
        return 0.5  # Default medium risk
```

### 2. Defense in Depth Strategy

```python
class DefenseInDepthSecurity:
    """Implement multiple layers of security"""
    
    def __init__(self):
        self.layers = [
            NetworkSecurityLayer(),
            ApplicationSecurityLayer(),
            DataSecurityLayer(),
            IdentitySecurityLayer()
        ]
    
    def process_request(self, request):
        """Process request through all security layers"""
        context = SecurityContext(request)
        
        for layer in self.layers:
            try:
                context = layer.process(context)
                if context.blocked:
                    return context.create_response()
            except SecurityException as e:
                return self.create_error_response(e)
        
        return context.create_success_response()

class NetworkSecurityLayer:
    """Network-level security controls"""
    
    def process(self, context):
        # IP allowlist/blocklist
        if context.source_ip in self.get_blocked_ips():
            context.block("IP address blocked")
        
        # Rate limiting by IP
        if self.is_rate_limited(context.source_ip):
            context.block("Rate limit exceeded")
        
        # DDoS protection
        if self.detect_ddos_pattern(context):
            context.block("DDoS pattern detected")
        
        return context

class ApplicationSecurityLayer:
    """Application-level security controls"""
    
    def process(self, context):
        # Input validation
        if not self.validate_input(context.request_data):
            context.block("Invalid input detected")
        
        # SQL injection detection
        if self.detect_sql_injection(context.request_data):
            context.block("SQL injection attempt")
        
        # XSS detection
        if self.detect_xss(context.request_data):
            context.block("XSS attempt detected")
        
        return context

class DataSecurityLayer:
    """Data-level security controls"""
    
    def process(self, context):
        # Data classification
        context.data_classification = self.classify_data(context.requested_resource)
        
        # Encryption requirements
        if context.data_classification == 'sensitive' and not context.encrypted_channel:
            context.block("Encryption required for sensitive data")
        
        # Data loss prevention
        if self.detect_data_exfiltration(context):
            context.block("Data exfiltration attempt")
        
        return context

class IdentitySecurityLayer:
    """Identity and access management"""
    
    def process(self, context):
        # Multi-factor authentication
        if context.requires_mfa and not context.mfa_verified:
            context.block("MFA required")
        
        # Privileged access monitoring
        if context.is_privileged_operation:
            self.log_privileged_access(context)
        
        # Session validation
        if not self.validate_session(context.session_token):
            context.block("Invalid session")
        
        return context
```

### 3. Secure Configuration Management

```python
import os
from cryptography.fernet import Fernet
import boto3
from azure.keyvault.secrets import SecretClient

class SecureConfigManager:
    """Manage sensitive configuration securely"""
    
    def __init__(self, config_source='env'):
        self.config_source = config_source
        self.encryption_key = self.get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def get_secret(self, secret_name, default=None):
        """Get secret from secure storage"""
        if self.config_source == 'aws_secrets':
            return self.get_aws_secret(secret_name, default)
        elif self.config_source == 'azure_keyvault':
            return self.get_azure_secret(secret_name, default)
        elif self.config_source == 'env_encrypted':
            return self.get_encrypted_env(secret_name, default)
        else:
            return os.getenv(secret_name, default)
    
    def get_aws_secret(self, secret_name, default=None):
        """Get secret from AWS Secrets Manager"""
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        except Exception:
            return default
    
    def get_azure_secret(self, secret_name, default=None):
        """Get secret from Azure Key Vault"""
        try:
            credential = DefaultAzureCredential()
            vault_url = os.getenv('AZURE_KEYVAULT_URL')
            client = SecretClient(vault_url=vault_url, credential=credential)
            secret = client.get_secret(secret_name)
            return secret.value
        except Exception:
            return default
    
    def get_encrypted_env(self, secret_name, default=None):
        """Get encrypted environment variable"""
        encrypted_value = os.getenv(f"{secret_name}_ENCRYPTED")
        if encrypted_value:
            try:
                return self.cipher_suite.decrypt(encrypted_value.encode()).decode()
            except Exception:
                pass
        return default
    
    def rotate_secrets(self):
        """Implement secret rotation"""
        secrets_to_rotate = [
            'DATABASE_PASSWORD',
            'JWT_SECRET_KEY',
            'API_ENCRYPTION_KEY'
        ]
        
        for secret in secrets_to_rotate:
            new_value = self.generate_secure_value()
            self.update_secret(secret, new_value)
            self.notify_services_of_rotation(secret)

# Configuration with security defaults
class SecurityConfig:
    def __init__(self):
        self.config_manager = SecureConfigManager()
        
        # Database security
        self.db_connection_params = {
            'host': self.config_manager.get_secret('DB_HOST'),
            'password': self.config_manager.get_secret('DB_PASSWORD'),
            'sslmode': 'require',
            'sslcert': '/path/to/client-cert.pem',
            'sslkey': '/path/to/client-key.pem',
            'sslrootcert': '/path/to/ca-cert.pem'
        }
        
        # JWT security
        self.jwt_config = {
            'secret_key': self.config_manager.get_secret('JWT_SECRET_KEY'),
            'algorithm': 'RS256',  # Use asymmetric encryption
            'access_token_expires': 900,  # 15 minutes
            'refresh_token_expires': 604800,  # 7 days
            'issuer': 'your-service-name',
            'audience': 'your-api-clients'
        }
        
        # TLS/SSL configuration
        self.tls_config = {
            'min_version': 'TLSv1.2',
            'ciphers': [
                'ECDHE+AESGCM',
                'ECDHE+CHACHA20',
                'DHE+AESGCM',
                'DHE+CHACHA20',
                '!aNULL',
                '!MD5',
                '!DSS'
            ],
            'cert_file': '/path/to/server-cert.pem',
            'key_file': '/path/to/server-key.pem'
        }
```

## Security Monitoring and Incident Response

### 1. Security Event Logging

```python
import logging
import json
from datetime import datetime
import hashlib

class SecurityLogger:
    """Centralized security event logging"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
        self.setup_structured_logging()
    
    def setup_structured_logging(self):
        """Configure structured logging for security events"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_authentication_event(self, user_id, event_type, success, details=None):
        """Log authentication events"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'authentication',
            'event_category': event_type,
            'user_id': self.hash_pii(user_id),
            'success': success,
            'source_ip': self.get_client_ip(),
            'user_agent': self.get_user_agent(),
            'details': details or {}
        }
        
        if success:
            self.logger.info(json.dumps(event))
        else:
            self.logger.warning(json.dumps(event))
    
    def log_authorization_event(self, user_id, resource, action, granted, reason=None):
        """Log authorization events"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'authorization',
            'user_id': self.hash_pii(user_id),
            'resource': resource,
            'action': action,
            'granted': granted,
            'reason': reason,
            'source_ip': self.get_client_ip()
        }
        
        if granted:
            self.logger.info(json.dumps(event))
        else:
            self.logger.warning(json.dumps(event))
    
    def log_security_incident(self, incident_type, severity, description, affected_resources=None):
        """Log security incidents"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'security_incident',
            'incident_type': incident_type,
            'severity': severity,
            'description': description,
            'affected_resources': affected_resources or [],
            'source_ip': self.get_client_ip(),
            'incident_id': self.generate_incident_id()
        }
        
        if severity in ['high', 'critical']:
            self.logger.error(json.dumps(event))
        else:
            self.logger.warning(json.dumps(event))
    
    def hash_pii(self, value):
        """Hash personally identifiable information"""
        if not value:
            return None
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]

# Security monitoring middleware
security_logger = SecurityLogger()

@app.before_request
def log_request():
    """Log security-relevant requests"""
    if request.path.startswith('/admin'):
        security_logger.log_authorization_event(
            user_id=getattr(request, 'current_user', {}).get('user_id'),
            resource=request.path,
            action=request.method,
            granted=True,  # Will be updated in after_request if denied
            reason="Admin area access"
        )

@app.after_request
def log_response(response):
    """Log security-relevant responses"""
    if response.status_code == 401:
        security_logger.log_authentication_event(
            user_id=getattr(request, 'current_user', {}).get('user_id'),
            event_type='authentication_failure',
            success=False,
            details={'endpoint': request.path}
        )
    elif response.status_code == 403:
        security_logger.log_authorization_event(
            user_id=getattr(request, 'current_user', {}).get('user_id'),
            resource=request.path,
            action=request.method,
            granted=False,
            reason="Insufficient permissions"
        )
    
    return response
```

### 2. Anomaly Detection

```python
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta

class SecurityAnomalyDetector:
    """Detect security anomalies in real-time"""
    
    def __init__(self):
        self.request_patterns = defaultdict(lambda: deque(maxlen=1000))
        self.user_behavior = defaultdict(lambda: {
            'normal_hours': set(),
            'common_ips': set(),
            'typical_endpoints': defaultdict(int)
        })
        self.baseline_metrics = {}
    
    def analyze_request_pattern(self, user_id, request_data):
        """Analyze request patterns for anomalies"""
        current_time = datetime.utcnow()
        
        anomalies = []
        
        # Check for unusual request frequency
        if self.detect_frequency_anomaly(user_id, current_time):
            anomalies.append({
                'type': 'high_frequency',
                'description': 'Unusually high request frequency',
                'severity': 'medium'
            })
        
        # Check for unusual access times
        if self.detect_time_anomaly(user_id, current_time):
            anomalies.append({
                'type': 'unusual_time',
                'description': 'Access outside normal hours',
                'severity': 'low'
            })
        
        # Check for unusual IP addresses
        if self.detect_ip_anomaly(user_id, request_data.get('source_ip')):
            anomalies.append({
                'type': 'unusual_ip',
                'description': 'Access from unknown IP address',
                'severity': 'medium'
            })
        
        # Check for privilege escalation attempts
        if self.detect_privilege_escalation(user_id, request_data):
            anomalies.append({
                'type': 'privilege_escalation',
                'description': 'Potential privilege escalation attempt',
                'severity': 'high'
            })
        
        return anomalies
    
    def detect_frequency_anomaly(self, user_id, current_time):
        """Detect unusual request frequency"""
        user_requests = self.request_patterns[user_id]
        user_requests.append(current_time)
        
        # Count requests in last minute
        minute_ago = current_time - timedelta(minutes=1)
        recent_requests = sum(1 for req_time in user_requests if req_time > minute_ago)
        
        # Dynamic threshold based on user's historical behavior
        normal_rate = self.calculate_normal_rate(user_id)
        threshold = normal_rate * 3  # 3x normal rate
        
        return recent_requests > threshold
    
    def detect_time_anomaly(self, user_id, current_time):
        """Detect access outside normal hours"""
        user_profile = self.user_behavior[user_id]
        current_hour = current_time.hour
        
        if not user_profile['normal_hours']:
            # Learning phase - record normal hours
            user_profile['normal_hours'].add(current_hour)
            return False
        
        # Check if current hour is significantly different
        normal_hours = user_profile['normal_hours']
        if len(normal_hours) < 5:  # Still learning
            user_profile['normal_hours'].add(current_hour)
            return False
        
        # Anomaly if accessing outside established pattern
        hour_range = range(min(normal_hours), max(normal_hours) + 1)
        return current_hour not in hour_range
    
    def detect_privilege_escalation(self, user_id, request_data):
        """Detect potential privilege escalation"""
        endpoint = request_data.get('endpoint', '')
        user_role = request_data.get('user_role', '')
        
        # Check for admin endpoint access by non-admin users
        admin_endpoints = ['/admin', '/manage', '/system', '/config']
        if any(admin_ep in endpoint for admin_ep in admin_endpoints):
            if user_role not in ['admin', 'super_admin']:
                return True
        
        # Check for rapid role changes
        recent_roles = self.get_recent_roles(user_id)
        if len(set(recent_roles)) > 2:  # Multiple role changes
            return True
        
        return False

class ThreatIntelligence:
    """Integrate with threat intelligence feeds"""
    
    def __init__(self):
        self.malicious_ips = set()
        self.malicious_domains = set()
        self.attack_signatures = []
        self.update_threat_feeds()
    
    def update_threat_feeds(self):
        """Update threat intelligence data"""
        # In production, integrate with threat intel APIs
        # Example: VirusTotal, AlienVault OTX, etc.
        pass
    
    def check_ip_reputation(self, ip_address):
        """Check IP against threat intelligence"""
        if ip_address in self.malicious_ips:
            return {
                'malicious': True,
                'threat_type': 'known_bad_ip',
                'confidence': 0.9
            }
        
        # Check against additional sources
        return {'malicious': False}
    
    def detect_attack_patterns(self, request_data):
        """Detect known attack patterns"""
        payload = str(request_data)
        
        attack_patterns = [
            (r'<script.*?>.*?</script>', 'xss_attempt'),
            (r'union.*select', 'sql_injection'),
            (r'\.\./', 'directory_traversal'),
            (r'eval\(.*\)', 'code_injection')
        ]
        
        detected_attacks = []
        for pattern, attack_type in attack_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                detected_attacks.append({
                    'type': attack_type,
                    'confidence': 0.8,
                    'pattern': pattern
                })
        
        return detected_attacks
```

## Learning Objectives

By the end of this section, you should be able to:

- **Implement comprehensive authentication and authorization** systems using JWT, OAuth 2.0, and OpenID Connect
- **Design secure API architectures** with proper input validation, rate limiting, and security headers
- **Configure secure service-to-service communication** using mTLS, service mesh, and API gateways
- **Apply security patterns** including Zero Trust, Defense in Depth, and Principle of Least Privilege
- **Implement security monitoring and incident response** with proper logging, anomaly detection, and threat intelligence
- **Secure microservices infrastructure** including container security, network policies, and secrets management
- **Perform security testing and validation** using automated tools and manual penetration testing techniques

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Implement JWT-based authentication with proper token validation and refresh  
□ Set up OAuth 2.0 flows for different client types (web, mobile, service-to-service)  
□ Configure mTLS for secure service communication  
□ Design and implement API rate limiting and throttling  
□ Set up comprehensive security monitoring and alerting  
□ Implement input validation and output encoding to prevent injection attacks  
□ Configure service mesh security policies (Istio, Linkerd)  
□ Design incident response procedures for security breaches  
□ Perform threat modeling for microservices architecture  
□ Implement secrets management with rotation and auditing  

### Practical Exercises

**Exercise 1: JWT Implementation**
```python
# TODO: Implement a complete JWT authentication system
# Include: token generation, validation, refresh, and blacklisting
class JWTManager:
    def __init__(self):
        pass
    
    def generate_tokens(self, user_data):
        # Your implementation here
        pass
    
    def validate_token(self, token):
        # Your implementation here
        pass
    
    def refresh_token(self, refresh_token):
        # Your implementation here
        pass
```

**Exercise 2: API Security Implementation**
```python
# TODO: Create a secure API endpoint with comprehensive protection
@app.route('/api/sensitive-data')
def get_sensitive_data():
    # Implement:
    # - Authentication check
    # - Authorization verification  
    # - Input validation
    # - Rate limiting
    # - Security headers
    # - Audit logging
    pass
```

**Exercise 3: Service Mesh Security**
```yaml
# TODO: Configure Istio security policies
# Create policies for:
# - mTLS enforcement
# - Service-to-service authorization
# - Network traffic restrictions
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: your-policy
# Your configuration here
```

## Study Materials

### Essential Reading
- **Primary:** "Microservices Security in Action" by Prabath Siriwardena and Nuwan Dias
- **OAuth 2.0:** "OAuth 2.0 in Action" by Justin Richer and Antonio Sanso  
- **API Security:** "API Security in Action" by Neil Madden
- **Cloud Security:** "Cloud Security and Privacy" by Tim Mather

### Online Resources
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OAuth 2.0 Security Best Current Practice](https://tools.ietf.org/html/draft-ietf-oauth-security-topics)
- [JWT Security Best Practices](https://auth0.com/blog/a-look-at-the-latest-draft-for-jwt-bcp/)

### Tools and Frameworks
- **Authentication:** Auth0, Keycloak, AWS Cognito
- **API Testing:** Postman, OWASP ZAP, Burp Suite
- **Service Mesh:** Istio, Linkerd, Consul Connect
- **Monitoring:** ELK Stack, Prometheus, Grafana, Jaeger

### Hands-on Labs
1. **JWT Lab:** Build a complete authentication system with refresh tokens
2. **OAuth Lab:** Implement OAuth 2.0 authorization server and client
3. **mTLS Lab:** Set up mutual TLS between services
4. **Security Testing Lab:** Perform penetration testing on your APIs
5. **Incident Response Lab:** Simulate and respond to security incidents

### Practice Scenarios

**Scenario 1: Financial Services API**
Design security for a banking API that handles:
- Account information access
- Money transfers
- Transaction history
- Regulatory compliance (PCI DSS, SOX)

**Scenario 2: Healthcare Microservices**
Implement security for healthcare services requiring:
- HIPAA compliance
- Patient data protection
- Role-based access (doctors, nurses, patients)
- Audit trails for all data access

**Scenario 3: E-commerce Platform**
Secure an e-commerce platform with:
- Customer authentication
- Payment processing security
- Inventory management
- Order tracking and privacy

### Certification Paths
- **Certified Information Systems Security Professional (CISSP)**
- **Certified Ethical Hacker (CEH)**
- **AWS Certified Security - Specialty**
- **Certified Information Security Manager (CISM)**

### Development Environment Setup

```bash
# Install security testing tools
pip install flask flask-jwt-extended authlib marshmallow
pip install cryptography pycryptodome
pip install requests oauthlib

# Docker for service mesh testing
docker pull istio/pilot:latest
docker pull prom/prometheus:latest

# Security scanning tools
npm install -g retire
pip install bandit safety
```

### Security Checklists

**Pre-deployment Security Checklist:**
□ All secrets stored securely (not in code)  
□ TLS/SSL properly configured  
□ Input validation implemented  
□ Rate limiting configured  
□ Security headers added  
□ Logging and monitoring enabled  
□ Vulnerability scanning completed  
□ Penetration testing performed  
□ Incident response plan documented  
□ Backup and recovery procedures tested  
