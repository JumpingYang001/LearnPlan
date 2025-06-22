# GPU Architecture Fundamentals

## Description
Understand modern GPU architecture (NVIDIA, AMD), CUDA cores, tensor cores, streaming multiprocessors, memory hierarchy, cache structure, and GPU execution model.

## Example Code
```cpp
// Example: Querying GPU properties with CUDA
#include <cuda_runtime.h>
#include <iostream>
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU Name: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem << std::endl;
    return 0;
}
```
