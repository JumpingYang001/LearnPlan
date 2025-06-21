# AMD ROCm Ecosystem

## Topics
- AMD's open compute platform
- HIP programming model
- MIOpen for deep learning
- ROCm-accelerated applications

### Example: Check ROCm Devices (Python)
```python
import torch
print('ROCm available:', torch.version.hip is not None)
```
