# CUDA Optimization Techniques

*Duration: 2 weeks*

## Overview

This section focuses on advanced optimization techniques for CUDA applications. You'll learn how to analyze and optimize memory access patterns, minimize thread divergence, maximize occupancy, and implement efficient synchronization mechanisms.

## Learning Objectives

By the end of this section, you will be able to:
- Optimize memory access patterns for maximum bandwidth
- Use shared memory effectively to reduce global memory traffic
- Minimize thread divergence impact on performance
- Maximize GPU occupancy for better resource utilization
- Implement efficient synchronization strategies

## 1. Memory Coalescing

### Understanding Memory Coalescing

Memory coalescing occurs when threads in a warp access consecutive memory locations, allowing the hardware to combine multiple memory requests into fewer transactions.

```cpp
// Bad: Non-coalesced access pattern
__global__ void nonCoalescedAccess(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Threads access memory with large strides
        output[idx] = input[idx * stride];
    }
}

// Good: Coalesced access pattern
__global__ void coalescedAccess(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Consecutive threads access consecutive memory locations
        output[idx] = input[idx];
    }
}

// Matrix transpose optimization
__global__ void naiveTranspose(float* input, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int input_idx = row * width + col;
        int output_idx = col * height + row;  // Non-coalesced write
        output[output_idx] = input[input_idx];
    }
}

__global__ void optimizedTranspose(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Coalesced read from input
    if (row < height && col < width) {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    
    __syncthreads();
    
    // Calculate transposed coordinates
    col = blockIdx.y * blockDim.y + threadIdx.x;
    row = blockIdx.x * blockDim.x + threadIdx.y;
    
    // Coalesced write to output
    if (row < width && col < height) {
        output[row * height + col] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Strided Access Optimization

```cpp
// Problem: Processing every Nth element
__global__ void stridedAccessBad(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < n) {
        data[idx * stride] *= 2.0f;  // Poor memory access pattern
    }
}

// Solution: Restructure algorithm to use consecutive access
__global__ void stridedAccessGood(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread with consecutive access
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        if (i % stride == 0) {
            data[i] *= 2.0f;
        }
    }
}

// Structure of Arrays (SoA) vs Array of Structures (AoS)
struct Particle_AoS {
    float x, y, z;
    float vx, vy, vz;
};

// AoS - poor for SIMD operations
__global__ void updateParticles_AoS(Particle_AoS* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles[idx].x += particles[idx].vx;  // Scattered memory access
        particles[idx].y += particles[idx].vy;
        particles[idx].z += particles[idx].vz;
    }
}

// SoA - better for SIMD operations
__global__ void updateParticles_SoA(float* x, float* y, float* z,
                                   float* vx, float* vy, float* vz, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += vx[idx];  // Coalesced memory access
        y[idx] += vy[idx];
        z[idx] += vz[idx];
    }
}
```

### Memory Alignment and Padding

```cpp
// Ensure proper memory alignment
void* alignedMalloc(size_t size, size_t alignment) {
    void* ptr;
    cudaMallocHost(&ptr, size + alignment - 1);
    
    // Align pointer
    uintptr_t addr = (uintptr_t)ptr;
    addr = (addr + alignment - 1) & ~(alignment - 1);
    
    return (void*)addr;
}

// Structure padding for optimal access
struct __align__(16) OptimizedStruct {
    float a, b, c, d;  // 16-byte aligned
};

// Benchmark memory access patterns
void benchmarkMemoryPatterns() {
    const int N = 1000000;
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // Initialize data
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Benchmark different access patterns
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Coalesced access
    cudaEventRecord(start);
    coalescedAccess<<<grid, block>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float coalescedTime;
    cudaEventElapsedTime(&coalescedTime, start, stop);
    
    // Non-coalesced access
    cudaEventRecord(start);
    nonCoalescedAccess<<<grid, block>>>(d_input, d_output, N, 32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float nonCoalescedTime;
    cudaEventElapsedTime(&nonCoalescedTime, start, stop);
    
    std::cout << "Coalesced access time: " << coalescedTime << " ms\n";
    std::cout << "Non-coalesced access time: " << nonCoalescedTime << " ms\n";
    std::cout << "Speedup: " << nonCoalescedTime / coalescedTime << "x\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

## 2. Shared Memory Usage

### Shared Memory Allocation Strategies

```cpp
// Static shared memory allocation
__global__ void staticSharedMemory() {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    shared_data[tid] = tid;
    
    __syncthreads();
    
    // Use shared_data...
}

// Dynamic shared memory allocation
__global__ void dynamicSharedMemory(float* global_data, int n) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < n) {
        shared_data[tid] = global_data[idx];
    }
    
    __syncthreads();
    
    // Process data in shared memory
    if (tid > 0 && tid < blockDim.x - 1) {
        float result = (shared_data[tid-1] + shared_data[tid] + shared_data[tid+1]) / 3.0f;
        if (idx < n) {
            global_data[idx] = result;
        }
    }
}

// Launch with dynamic shared memory
void launchWithSharedMemory() {
    const int N = 1024;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const size_t sharedMemSize = blockSize * sizeof(float);
    
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    dynamicSharedMemory<<<gridSize, blockSize, sharedMemSize>>>(d_data, N);
    
    cudaFree(d_data);
}
```

### Bank Conflicts

```cpp
// Understanding bank conflicts
__global__ void bankConflictExample() {
    __shared__ float shared_data[32][32];
    
    int tid = threadIdx.x;
    
    // No bank conflicts - different banks
    shared_data[tid][0] = tid;
    
    // Bank conflicts - same bank, different addresses
    shared_data[0][tid] = tid;  // All threads access bank 0
    
    // Avoiding bank conflicts with padding
    __shared__ float padded_data[32][33];  // +1 for padding
    padded_data[tid][tid] = tid;  // Now consecutive threads access different banks
}

// Matrix multiplication with shared memory optimization
__global__ void matmulSharedOptimized(float* A, float* B, float* C,
                                     int M, int N, int K) {
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Shared Memory vs L1 Cache Configuration

```cpp
// Configure shared memory vs L1 cache
void configureSharedMemory() {
    // Get current configuration
    cudaFuncCache config;
    cudaDeviceGetCacheConfig(&config);
    
    switch (config) {
        case cudaFuncCachePreferNone:
            std::cout << "No preference\n";
            break;
        case cudaFuncCachePreferShared:
            std::cout << "Prefer shared memory\n";
            break;
        case cudaFuncCachePreferL1:
            std::cout << "Prefer L1 cache\n";
            break;
        case cudaFuncCachePreferEqual:
            std::cout << "Equal shared memory and L1 cache\n";
            break;
    }
    
    // Set global configuration
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    
    // Set per-kernel configuration
    cudaFuncSetCacheConfig(matmulSharedOptimized, cudaFuncCachePreferShared);
}

// Bandwidth benchmark for shared memory
__global__ void sharedMemoryBandwidth(float* data, int iterations) {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    float value = data[blockIdx.x * blockDim.x + tid];
    
    // Shared memory operations
    for (int i = 0; i < iterations; i++) {
        shared_data[tid] = value;
        __syncthreads();
        value = shared_data[(tid + 1) % blockDim.x];
        __syncthreads();
    }
    
    data[blockIdx.x * blockDim.x + tid] = value;
}
```

## 3. Thread Divergence

### Understanding Warp Execution

```cpp
// Thread divergence example
__global__ void threadDivergenceExample(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // This creates divergence within warps
        if (data[idx] > 0) {
            data[idx] = data[idx] * 2;      // Some threads execute this
        } else {
            data[idx] = data[idx] + 1;      // Others execute this
        }
        // Threads reconverge here
        data[idx] += 10;  // All threads execute this
    }
}

// Minimizing divergence by restructuring conditions
__global__ void reduceDivergence(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int value = data[idx];
        
        // Use arithmetic operations instead of branches when possible
        int is_positive = (value > 0) ? 1 : 0;
        data[idx] = value * (1 + is_positive) + (1 - is_positive);
        data[idx] += 10;
    }
}

// Warp-level operations to handle divergence
__global__ void warpLevelOperations(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    
    if (idx < n) {
        int value = data[idx];
        
        // Check if any thread in warp needs special processing
        int mask = __ballot_sync(0xFFFFFFFF, value > 0);
        
        if (mask != 0) {  // At least one thread needs processing
            if (value > 0) {
                value *= 2;
            }
        }
        
        // All threads can now execute this
        data[idx] = value + 10;
    }
}
```

### Branch Optimization Techniques

```cpp
// Predication to avoid branches
__global__ void predicationExample(float* data, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Instead of: if (value > threshold) value *= 2.0f;
        // Use predication:
        float factor = (value > threshold) ? 2.0f : 1.0f;
        data[idx] = value * factor;
    }
}

// Loop optimization to reduce divergence
__global__ void loopOptimization(int* data, int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bad: Different loop counts cause divergence
    /*
    if (idx < n) {
        for (int i = 0; i < indices[idx]; i++) {
            data[idx] += i;
        }
    }
    */
    
    // Better: Uniform loop count with conditional execution
    if (idx < n) {
        int max_iterations = 100;  // Known maximum
        for (int i = 0; i < max_iterations; i++) {
            if (i < indices[idx]) {
                data[idx] += i;
            }
        }
    }
}

// Sort threads by execution path
__global__ void sortByExecutionPath(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int value = data[idx];
        
        // Group threads by condition using warp vote functions
        int positive_mask = __ballot_sync(0xFFFFFFFF, value > 0);
        int negative_mask = __ballot_sync(0xFFFFFFFF, value <= 0);
        
        // Process positive values first
        if (value > 0) {
            data[idx] = value * 2;
        }
        
        // Then process negative values
        if (value <= 0) {
            data[idx] = value + 1;
        }
    }
}
```

### Warp Vote Functions

```cpp
// Using warp vote functions for coordination
__global__ void warpVoteExample(int* data, int* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (idx < n) {
        int value = data[idx];
        bool condition = (value > 0);
        
        // Check if all threads in warp satisfy condition
        bool all_positive = __all_sync(0xFFFFFFFF, condition);
        
        // Check if any thread in warp satisfies condition
        bool any_positive = __any_sync(0xFFFFFFFF, condition);
        
        // Get ballot of threads satisfying condition
        unsigned int ballot = __ballot_sync(0xFFFFFFFF, condition);
        
        // Count number of threads satisfying condition
        int count = __popc(ballot);
        
        if (lane_id == 0) {  // First thread in warp reports results
            int warp_result_idx = blockIdx.x * (blockDim.x / 32) + warp_id;
            results[warp_result_idx * 4] = all_positive ? 1 : 0;
            results[warp_result_idx * 4 + 1] = any_positive ? 1 : 0;
            results[warp_result_idx * 4 + 2] = ballot;
            results[warp_result_idx * 4 + 3] = count;
        }
    }
}

// Warp shuffle for efficient communication
__global__ void warpShuffleReduction(float* data, float* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    float value = (idx < n) ? data[idx] : 0.0f;
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    
    // First thread in each warp stores result
    if (lane_id == 0) {
        int warp_result_idx = blockIdx.x * (blockDim.x / 32) + warp_id;
        results[warp_result_idx] = value;
    }
}
```

## 4. Occupancy Optimization

### Understanding Occupancy

```cpp
// Query occupancy information
void analyzeOccupancy() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "Registers per SM: " << prop.regsPerMultiprocessor << std::endl;
}

// Calculate theoretical occupancy
int calculateOccupancy(int blockSize, int sharedMemPerBlock, int regsPerThread) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Limits based on threads
    int maxBlocksPerSM_threads = prop.maxThreadsPerMultiProcessor / blockSize;
    
    // Limits based on blocks
    int maxBlocksPerSM_blocks = prop.maxBlocksPerMultiProcessor;
    
    // Limits based on shared memory
    int maxBlocksPerSM_shmem = prop.sharedMemPerMultiprocessor / sharedMemPerBlock;
    
    // Limits based on registers
    int maxBlocksPerSM_regs = prop.regsPerMultiprocessor / (blockSize * regsPerThread);
    
    // Actual occupancy is minimum of all limits
    int actualBlocks = std::min({maxBlocksPerSM_threads, maxBlocksPerSM_blocks,
                                maxBlocksPerSM_shmem, maxBlocksPerSM_regs});
    
    int occupancy = (actualBlocks * blockSize * 100) / prop.maxThreadsPerMultiProcessor;
    
    return occupancy;
}
```

### Register Usage Optimization

```cpp
// High register usage kernel (poor occupancy)
__global__ void highRegisterUsage(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Many local variables increase register usage
    float temp1, temp2, temp3, temp4, temp5;
    float temp6, temp7, temp8, temp9, temp10;
    
    if (idx < n) {
        float value = data[idx];
        
        // Complex computation using many registers
        temp1 = sin(value);
        temp2 = cos(value);
        temp3 = temp1 * temp2;
        temp4 = temp3 + value;
        temp5 = temp4 * temp4;
        temp6 = sqrt(temp5);
        temp7 = temp6 + temp1;
        temp8 = temp7 * temp2;
        temp9 = temp8 + temp3;
        temp10 = temp9 * temp4;
        
        data[idx] = temp10;
    }
}

// Optimized version with reduced register usage
__global__ void lowRegisterUsage(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Reuse variables to reduce register pressure
        float temp = sin(value);
        temp *= cos(value);
        temp += value;
        temp *= temp;
        temp = sqrt(temp);
        temp += sin(value);
        temp *= cos(value);
        
        data[idx] = temp;
    }
}

// Limit register usage with compiler directive
__global__ __launch_bounds__(256, 2)  // Max 256 threads, min 2 blocks per SM
void limitedRegisterKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Kernel implementation
        data[idx] = sqrt(data[idx] * data[idx] + 1.0f);
    }
}
```

### Block Size Selection

```cpp
// Automated block size selection
template<typename KernelFunc, typename... Args>
int findOptimalBlockSize(KernelFunc kernel, Args... args) {
    int minGridSize, blockSize;
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
    
    return blockSize;
}

// Test different block sizes
void blockSizeExperiment() {
    const int N = 1000000;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};
    
    for (int blockSize : blockSizes) {
        int gridSize = (N + blockSize - 1) / blockSize;
        
        cudaEventRecord(start);
        simpleKernel<<<gridSize, blockSize>>>(d_data, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time;
        cudaEventElapsedTime(&time, start, stop);
        
        int occupancy = calculateOccupancy(blockSize, 0, 16);
        
        std::cout << "Block size: " << blockSize 
                  << ", Time: " << time << " ms"
                  << ", Occupancy: " << occupancy << "%" << std::endl;
    }
    
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void simpleKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrt(data[idx] * data[idx] + 1.0f);
    }
}
```

### Shared Memory Allocation Optimization

```cpp
// Dynamic shared memory allocation based on block size
__global__ void adaptiveSharedMemory(float* data, int n) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory efficiently based on block size
    if (idx < n) {
        shared_data[tid] = data[idx];
    }
    
    __syncthreads();
    
    // Perform operations using shared memory
    if (tid > 0 && tid < blockDim.x - 1) {
        float result = (shared_data[tid-1] + shared_data[tid] + shared_data[tid+1]) / 3.0f;
        if (idx < n) {
            data[idx] = result;
        }
    }
}

// Launch with optimal shared memory usage
void launchAdaptiveKernel() {
    const int N = 100000;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Find optimal block size
    int optimalBlockSize = findOptimalBlockSize(adaptiveSharedMemory, d_data, N);
    int gridSize = (N + optimalBlockSize - 1) / optimalBlockSize;
    
    // Calculate shared memory size
    size_t sharedMemSize = optimalBlockSize * sizeof(float);
    
    adaptiveSharedMemory<<<gridSize, optimalBlockSize, sharedMemSize>>>(d_data, N);
    
    cudaFree(d_data);
}
```

## 5. Synchronization

### Block-Level Synchronization

```cpp
// Basic synchronization with __syncthreads()
__global__ void basicSynchronization(float* data, int n) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < n) {
        shared_data[tid] = data[idx];
    }
    
    __syncthreads();  // Ensure all threads have loaded data
    
    // Process data
    if (tid > 0 && tid < blockDim.x - 1) {
        float result = (shared_data[tid-1] + shared_data[tid] + shared_data[tid+1]) / 3.0f;
        shared_data[tid] = result;
    }
    
    __syncthreads();  // Ensure all processing is complete
    
    // Write back to global memory
    if (idx < n) {
        data[idx] = shared_data[tid];
    }
}

// Conditional synchronization
__global__ void conditionalSync(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int value = data[idx];
        
        // Only synchronize if needed
        if (__any_sync(0xFFFFFFFF, value < 0)) {
            __syncthreads();
            
            // Handle negative values collectively
            if (value < 0) {
                data[idx] = abs(value);
            }
        }
    }
}
```

### Atomic Operations

```cpp
// Basic atomic operations
__global__ void atomicOperationsExample(int* counters, float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Atomic increment
        if (value > 0) {
            atomicAdd(&counters[0], 1);
        }
        
        // Atomic compare and swap
        int old_val = atomicCAS(&counters[1], 0, 1);
        
        // Atomic exchange
        int prev_val = atomicExch(&counters[2], idx);
        
        // Atomic minimum/maximum
        atomicMin(&counters[3], (int)value);
        atomicMax(&counters[4], (int)value);
    }
}

// Custom atomic operations for floats
__device__ float atomicAddFloat(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed,
                       __float_as_uint(val + __uint_as_float(assumed)));
    } while (assumed != old);
    
    return __uint_as_float(old);
}

// Lock-free data structures
__global__ void lockFreeStack(int* stack, int* top, int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Push operation
        int value = data[idx];
        int old_top = atomicAdd(top, 1);
        stack[old_top] = value;
    }
}
```

### Cooperative Groups

```cpp
#include <cooperative_groups.h>
using namespace cooperative_groups;

// Thread block groups
__global__ void threadBlockGroups(float* data, int n) {
    thread_block block = this_thread_block();
    
    int idx = block.group_index().x * block.group_dim().x + block.thread_rank();
    
    if (idx < n) {
        data[idx] *= 2.0f;
    }
    
    block.sync();  // Equivalent to __syncthreads()
    
    // Use block-level collective operations
    float sum = block.reduce(data[block.thread_rank()], plus<float>());
    
    if (block.thread_rank() == 0) {
        // First thread in block has the sum
        data[block.group_index().x] = sum;
    }
}

// Warp-level groups
__global__ void warpLevelGroups(float* data, int n) {
    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Warp-level reduction
        float sum = warp.reduce(value, plus<float>());
        
        if (warp.thread_rank() == 0) {
            // First thread in warp stores result
            data[blockIdx.x * (blockDim.x / 32) + warp.meta_group_rank()] = sum;
        }
    }
}

// Grid-level synchronization (requires compatible hardware)
__global__ void gridLevelSync(float* data, int n) {
    grid_group grid = this_grid();
    
    int idx = grid.thread_rank();
    
    if (idx < n) {
        data[idx] *= 2.0f;
    }
    
    grid.sync();  // Synchronize entire grid
    
    // All blocks are now synchronized
    if (idx < n) {
        data[idx] += 1.0f;
    }
}
```

## Practical Exercises

1. **Memory Access Pattern Analysis**
   - Implement and benchmark different access patterns
   - Measure memory bandwidth for various scenarios
   - Optimize matrix operations for coalesced access

2. **Shared Memory Optimization**
   - Create kernels that effectively use shared memory
   - Eliminate bank conflicts in various scenarios
   - Compare shared memory vs global memory performance

3. **Thread Divergence Minimization**
   - Identify and fix divergent code patterns
   - Implement branch-free algorithms
   - Use warp vote functions for coordination

4. **Occupancy Analysis**
   - Analyze occupancy for different kernel configurations
   - Optimize register usage and shared memory allocation
   - Find optimal block sizes for various workloads

## Key Takeaways

- Memory coalescing is crucial for achieving high bandwidth
- Shared memory can provide orders of magnitude speedup when used correctly
- Thread divergence significantly impacts performance within warps
- Occupancy optimization requires balancing multiple resource constraints
- Proper synchronization ensures correctness without unnecessary overhead

## Next Steps

After mastering CUDA optimization techniques, proceed to:
- Advanced CUDA Programming
- CUDA Streams and Events
- Multi-GPU programming
- Performance profiling tools
