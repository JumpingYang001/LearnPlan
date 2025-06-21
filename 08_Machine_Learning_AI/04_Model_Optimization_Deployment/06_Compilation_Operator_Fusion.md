# Compilation and Operator Fusion

## Topics
- Intermediate representations for ML models
- Operator fusion and graph optimization
- Just-in-time compilation for ML
- Compiler optimizations for ML models

### Example: TorchScript JIT Compilation
```python
import torch
class MyModule(torch.nn.Module):
    def forward(self, x):
        return x * 2
scripted = torch.jit.script(MyModule())
print(scripted.code)
```
