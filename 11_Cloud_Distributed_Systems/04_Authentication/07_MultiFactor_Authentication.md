# Multi-factor Authentication (MFA)

## Concepts
- Combines two or more authentication factors

## Example: TOTP (Python, using pyotp)
```python
import pyotp
secret = pyotp.random_base32()
totp = pyotp.TOTP(secret)
print('Current OTP:', totp.now())
```
