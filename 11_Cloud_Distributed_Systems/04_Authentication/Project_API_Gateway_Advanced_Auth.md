# Project: API Gateway with Advanced Authentication

## Description
Develop an API gateway supporting multiple authentication methods, token translation, and security monitoring.

## Example: API Gateway Auth (Python Flask)
```python
from flask import Flask, request, jsonify
app = Flask(__name__)
API_KEYS = {'key1', 'key2'}
@app.route('/api', methods=['GET'])
def api():
    key = request.headers.get('X-API-Key')
    if key in API_KEYS:
        return jsonify(message='Access granted')
    return jsonify(error='Unauthorized'), 401
if __name__ == '__main__':
    app.run()
```
