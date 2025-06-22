# OpenID Connect

## Overview
- Identity layer on top of OAuth 2.0
- Provides ID tokens and user info

## Example: Decoding an ID Token (Python)
```python
import jwt
id_token = 'your_id_token_here'
# In production, always verify signature and claims!
decoded = jwt.decode(id_token, options={"verify_signature": False})
print(decoded)
```
