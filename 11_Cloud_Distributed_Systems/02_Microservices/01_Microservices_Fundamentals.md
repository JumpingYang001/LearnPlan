# Microservices Fundamentals

## Description
Understand the microservices architectural style, compare monolithic vs. microservices architectures, learn core principles and benefits, and study challenges and trade-offs.

## Example Code
```python
# Example: Simple Microservice using Flask
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Microservice!"

if __name__ == '__main__':
    app.run(port=5000)
```
