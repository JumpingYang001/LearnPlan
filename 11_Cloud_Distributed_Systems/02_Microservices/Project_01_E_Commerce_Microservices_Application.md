# Project 1: E-Commerce Microservices Application

## Description
Build a complete e-commerce system with multiple services: product catalog, shopping cart, payment, and order services. Implement an API gateway and service discovery mechanism.

## Example Code
```python
# Example: Product Service (Flask)
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/products')
def products():
    return jsonify([{"id": 1, "name": "Book"}])

if __name__ == '__main__':
    app.run(port=5001)
```

```python
# Example: API Gateway (simple proxy)
from flask import Flask, request, jsonify
import requests
app = Flask(__name__)

@app.route('/api/products')
def api_products():
    resp = requests.get('http://localhost:5001/products')
    return jsonify(resp.json())

if __name__ == '__main__':
    app.run(port=8000)
```
