# JSON Web Tokens (JWT)

## Structure
- Header, Payload, Signature

## Example: Creating a JWT (Python)
```python
import jwt
encoded = jwt.encode({'user': 'alice'}, 'secret', algorithm='HS256')
print(encoded)
```
