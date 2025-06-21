# cuDNN and cuBLAS Libraries

## Topics
- NVIDIA's deep learning libraries
- Optimized primitives for neural networks
- Integration with high-level frameworks
- Applications using cuDNN and cuBLAS

### Example: Fast Convolution (Python)
```python
import torch
import torch.nn as nn
x = torch.randn(1, 3, 224, 224).cuda()
conv = nn.Conv2d(3, 16, 3).cuda()
y = conv(x)
print('Output shape:', y.shape)
```
