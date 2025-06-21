# A/B Testing and Deployment Patterns

## Topics
- Canary deployments and blue-green deployments
- Shadow mode deployment
- A/B testing for ML models
- Safe deployment strategies

### Example: A/B Test Routing (FastAPI)
```python
from fastapi import FastAPI, Request
import random
app = FastAPI()

@app.post('/predict')
async def predict(request: Request):
    if random.random() < 0.5:
        # Route to model A
        return {"model": "A", "result": 0}
    else:
        # Route to model B
        return {"model": "B", "result": 1}
```
