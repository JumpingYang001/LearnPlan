# OAuth 2.0 Framework

## Roles and Flows
- Roles: Resource Owner, Client, Authorization Server, Resource Server
- Flows: Authorization Code, Implicit, Client Credentials

## Example: OAuth 2.0 Authorization Code Flow (Python, using requests)
```python
import requests
# This is a simplified example, real OAuth 2.0 flows require redirect handling
# and secure client secret management.
auth_url = 'https://auth.example.com/authorize'
token_url = 'https://auth.example.com/token'
client_id = 'your_client_id'
client_secret = 'your_client_secret'
redirect_uri = 'https://yourapp.com/callback'

# Step 1: User authorizes and you get an authorization code
# Step 2: Exchange code for token
code = 'received_authorization_code'
data = {
    'grant_type': 'authorization_code',
    'code': code,
    'redirect_uri': redirect_uri,
    'client_id': client_id,
    'client_secret': client_secret
}
response = requests.post(token_url, data=data)
print(response.json())
```
