# Project: Full-stack Application with TDD

## Description
Develop a web application using TDD for all layers. Implement both unit and integration tests. Document the TDD workflow.

## Example: Flask API with TDD (Python)
```python
# app.py
from flask import Flask, jsonify
app = Flask(__name__)
@app.route('/ping')
def ping():
    return jsonify({'message': 'pong'})

# test_app.py
import unittest
from app import app
class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
    def test_ping(self):
        response = self.client.get('/ping')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'message': 'pong'})
```
