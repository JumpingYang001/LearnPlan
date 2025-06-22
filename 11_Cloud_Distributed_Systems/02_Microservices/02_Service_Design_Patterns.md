# Service Design Patterns

## Description
Master Domain-Driven Design (DDD) concepts, bounded contexts, aggregates, service decomposition strategies, and implement service design patterns.

## Example Code
```python
# Example: Service Decomposition
# Product Service and Order Service communicate via REST
# Product Service
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/products')
def get_products():
    return jsonify([{"id": 1, "name": "Book"}])

if __name__ == '__main__':
    app.run(port=5001)
```
