# CUDA Programming Model

## Topics
- CUDA architecture and execution model
- CUDA threads, blocks, and grids
- Memory hierarchy and optimization
- Basic CUDA kernels for ML operations

### Example: CUDA Kernel (C++)
```cpp
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```
