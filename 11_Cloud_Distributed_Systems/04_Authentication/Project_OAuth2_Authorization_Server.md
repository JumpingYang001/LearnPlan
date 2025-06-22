# Project: OAuth 2.0 Authorization Server

## Description
Build a complete OAuth 2.0 authorization server supporting multiple grant types, token management, and validation.

## Example: Minimal OAuth 2.0 Server (Python Flask)
```python
from flask import Flask, request, jsonify
app = Flask(__name__)
TOKENS = {}
@app.route('/token', methods=['POST'])
def token():
    grant_type = request.form.get('grant_type')
    if grant_type == 'client_credentials':
        client_id = request.form.get('client_id')
        client_secret = request.form.get('client_secret')
        # Validate client credentials (hardcoded for demo)
        if client_id == 'demo' and client_secret == 'secret':
            token = 'access_token_123'
            TOKENS[token] = client_id
            return jsonify(access_token=token, token_type='bearer')
    return jsonify(error='invalid_grant'), 400
if __name__ == '__main__':
    app.run()
```
