# Model Optimization Fundamentals

## Topics
- Need for model optimization
- Common bottlenecks in ML systems
- Tradeoff space (accuracy, latency, size, energy)
- ML deployment lifecycle

### Example: Measuring Model Latency (PyTorch)
```python
import torch
import time
model = ...  # your model
input = torch.randn(1, 3, 224, 224)
start = time.time()
with torch.no_grad():
    output = model(input)
print('Latency:', time.time() - start, 'seconds')
```
