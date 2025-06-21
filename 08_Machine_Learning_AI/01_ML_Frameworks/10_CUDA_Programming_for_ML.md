# CUDA Programming for ML

## CUDA Basics for ML
- GPU architecture for ML
- CUDA programming model
- Memory hierarchy
- Kernel optimization

## cuDNN Library
- Deep learning primitives
- Convolution algorithms
- Tensor operations
- Integration with frameworks

## NCCL (NVIDIA Collective Communications Library)
- Multi-GPU communication
- All-reduce operations
- Topology awareness
- Distributed training integration

## Custom CUDA Kernels
- Kernel development for ML
- Operation fusion
- Memory access patterns
- Performance profiling


### Example: Simple CUDA Kernel (C++)
```cpp
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

### Example: PyTorch CUDA Tensor (Python)
```python
import torch
a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = a + b
```
