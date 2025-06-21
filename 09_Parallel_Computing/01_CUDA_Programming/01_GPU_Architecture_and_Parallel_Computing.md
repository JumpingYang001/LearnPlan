# GPU Architecture and Parallel Computing

*Duration: 1 week*

## Overview

Understanding GPU architecture is fundamental to effective CUDA programming. This section covers the key differences between CPU and GPU architectures, NVIDIA's GPU architecture, and essential parallel computing concepts.

## Learning Objectives

By the end of this section, you will understand:
- The architectural differences between CPUs and GPUs
- NVIDIA's GPU architecture components
- Key parallel computing concepts and their application to GPU programming

## 1. GPU vs. CPU Architecture

### Architectural Differences

**CPU (Central Processing Unit):**
- Optimized for sequential task performance
- Complex control logic and large caches
- Few cores (typically 4-64) with high per-core performance
- Branch prediction and out-of-order execution
- Designed for latency optimization

**GPU (Graphics Processing Unit):**
- Optimized for parallel task execution
- Simple control logic with smaller caches per core
- Thousands of cores with lower per-core performance
- Massive parallelism and throughput optimization
- Designed for bandwidth optimization

### Example: Comparing Processing Approaches

```cpp
// CPU approach - Sequential processing
void cpu_vector_add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // One operation at a time
    }
}

// GPU approach - Parallel processing concept
// Each thread handles one element
__global__ void gpu_vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[i] = a[i] + b[i];  // Thousands of operations simultaneously
    }
}
```

### SIMD vs. SIMT Execution Models

**SIMD (Single Instruction, Multiple Data):**
- Traditional vector processing
- All processing units execute the same instruction
- Rigid synchronization

**SIMT (Single Instruction, Multiple Thread):**
- NVIDIA's approach for GPU execution
- Groups of threads (warps) execute the same instruction
- More flexible than SIMD - threads can diverge
- Automatic handling of control flow

```cpp
// Example showing SIMT flexibility
__global__ void simt_example(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] > 0) {
            data[idx] *= 2;      // Some threads execute this
        } else {
            data[idx] = 0;       // Others execute this
        }
    }
    // Threads reconverge here
}
```

### Memory Hierarchy

**CPU Memory Hierarchy:**
- Large L1, L2, L3 caches
- Complex cache coherency protocols
- Lower memory bandwidth
- Higher memory latency tolerance

**GPU Memory Hierarchy:**
- Smaller caches per core
- Higher memory bandwidth
- Lower latency tolerance
- Multiple memory types optimized for different access patterns

### Throughput vs. Latency Optimization

```cpp
// CPU optimization example - minimize latency
void cpu_optimized_sum(float* array, int n) {
    float sum = 0.0f;
    // Loop unrolling to reduce branch overhead
    for (int i = 0; i < n; i += 4) {
        sum += array[i] + array[i+1] + array[i+2] + array[i+3];
    }
}

// GPU optimization example - maximize throughput
__global__ void gpu_optimized_sum(float* array, float* result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? array[idx] : 0.0f;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}
```

## 2. NVIDIA GPU Architecture

### Streaming Multiprocessors (SMs)

SMs are the core computational units of NVIDIA GPUs:
- Each SM contains multiple CUDA cores
- Shared memory and register file
- Warp schedulers and dispatch units
- Special function units (SFUs)

```cpp
// Query SM properties
void query_sm_properties() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("CUDA Cores per SM: %d\n", 
           _ConvertSMVer2Cores(prop.major, prop.minor));
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
}
```

### CUDA Cores

- Basic arithmetic logic units (ALUs)
- Execute floating-point and integer operations
- Number varies by GPU architecture (e.g., 64, 128, or 192 per SM)

### Tensor Cores (Modern GPUs)

- Specialized units for mixed-precision matrix operations
- Accelerate AI/ML workloads
- Support for various data types (FP16, INT8, etc.)

```cpp
// Example using Tensor Cores via WMMA API
#include <mma.h>
using namespace nvcuda;

__global__ void tensor_core_example() {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Load matrices
    wmma::load_matrix_sync(a_frag, a_matrix, 16);
    wmma::load_matrix_sync(b_frag, b_matrix, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Perform matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result
    wmma::store_matrix_sync(c_matrix, c_frag, 16, wmma::mem_row_major);
}
```

### Memory Types

1. **Global Memory**
   - Largest capacity, highest latency
   - Accessible by all threads
   - Cached in L1/L2

2. **Shared Memory**
   - Fast, low-latency memory
   - Shared within a thread block
   - Configurable with L1 cache

3. **Constant Memory**
   - Read-only, cached
   - Broadcast efficiently to all threads
   - Limited size (64KB)

4. **Texture Memory**
   - Cached, optimized for spatial locality
   - Hardware interpolation
   - Good for image processing

```cpp
// Memory type examples
__constant__ float const_data[256];  // Constant memory

__global__ void memory_types_example(float* global_mem) {
    extern __shared__ float shared_mem[];  // Shared memory
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global to shared memory
    shared_mem[tid] = global_mem[idx];
    __syncthreads();
    
    // Use constant memory
    float result = shared_mem[tid] * const_data[tid % 256];
    
    // Write back to global memory
    global_mem[idx] = result;
}
```

### Warp Execution Model

- Warp: Group of 32 threads executing in lockstep
- Basic scheduling unit
- All threads in a warp execute the same instruction
- Thread divergence reduces efficiency

```cpp
// Example showing warp behavior
__global__ void warp_example(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / 32;
    int lane_id = idx % 32;
    
    if (idx < n) {
        // All threads in warp execute together
        if (lane_id == 0) {
            printf("Warp %d processing\n", warp_id);
        }
        
        // Warp-level primitive example
        int value = data[idx];
        int sum = __shfl_down_sync(0xFFFFFFFF, value, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
        
        if (lane_id == 0) {
            // First thread in warp has the sum
            data[warp_id] = sum;
        }
    }
}
```

## 3. Parallel Computing Concepts

### Amdahl's Law

Amdahl's Law describes the theoretical speedup from parallelization:

**Speedup = 1 / (S + P/N)**

Where:
- S = Sequential portion of the program
- P = Parallel portion of the program
- N = Number of processors

```cpp
// Example demonstrating Amdahl's Law
void amdahl_example() {
    const int N = 1000000;
    float* data = new float[N];
    
    // Sequential initialization (10% of total time)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }
    auto init_time = std::chrono::high_resolution_clock::now() - start;
    
    // Parallel computation (90% of total time)
    start = std::chrono::high_resolution_clock::now();
    // This part can be parallelized
    for (int i = 0; i < N; i++) {
        data[i] = sqrt(data[i] * data[i] + 1.0f);
    }
    auto compute_time = std::chrono::high_resolution_clock::now() - start;
    
    // Maximum theoretical speedup with infinite processors:
    // 1 / (0.1 + 0.9/∞) = 10x
    
    delete[] data;
}
```

### Task and Data Parallelism

**Task Parallelism:**
- Different tasks executed simultaneously
- Tasks may operate on different data

**Data Parallelism:**
- Same operation applied to different data elements
- More suitable for GPU acceleration

```cpp
// Task parallelism example (better for CPU)
void task_parallel_example() {
    std::thread t1([]() { /* Task 1: Image filtering */ });
    std::thread t2([]() { /* Task 2: Audio processing */ });
    std::thread t3([]() { /* Task 3: Network I/O */ });
    
    t1.join(); t2.join(); t3.join();
}

// Data parallelism example (ideal for GPU)
__global__ void data_parallel_example(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Same operation on different data elements
        output[idx] = sqrt(input[idx] * input[idx] + 1.0f);
    }
}
```

### Decomposition Strategies

1. **Domain Decomposition**
   - Divide data into chunks
   - Each processor works on a chunk

2. **Functional Decomposition**
   - Divide algorithm into stages
   - Pipeline processing

```cpp
// Domain decomposition example
__global__ void domain_decomposition(float* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // Each thread processes one matrix element
        matrix[idx] = process_element(matrix[idx]);
    }
}

// Functional decomposition example
__global__ void stage1_filter(float* input, float* intermediate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        intermediate[idx] = apply_filter(input[idx]);
    }
}

__global__ void stage2_transform(float* intermediate, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = apply_transform(intermediate[idx]);
    }
}
```

### Synchronization and Communication

GPU synchronization mechanisms:
- `__syncthreads()` - Block-level synchronization
- `__syncwarp()` - Warp-level synchronization
- `cudaDeviceSynchronize()` - Host-device synchronization
- Atomic operations for coordination

```cpp
__global__ void synchronization_example(int* data, int* result, int n) {
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    if (idx < n) {
        shared_data[tid] = data[idx];
    } else {
        shared_data[tid] = 0;
    }
    
    // Synchronize all threads in block
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();  // Synchronize each iteration
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);  // Atomic operation
    }
}
```

### Race Conditions and Deadlocks

**Race Conditions:**
- Multiple threads accessing shared data without proper synchronization
- Results depend on thread execution order

**Deadlocks:**
- Less common in GPU programming due to SIMT execution
- Can occur with complex synchronization patterns

```cpp
// Race condition example (incorrect)
__global__ void race_condition_bad(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        (*counter)++;  // Race condition! Multiple threads modifying counter
    }
}

// Correct version using atomic operations
__global__ void race_condition_fixed(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(counter, 1);  // Atomic operation prevents race condition
    }
}

// Avoiding deadlock in cooperative groups
__global__ void avoid_deadlock(int* data, int n) {
    cooperative_groups::thread_block block = 
        cooperative_groups::this_thread_block();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Process data
        data[idx] = process(data[idx]);
    }
    
    // Safe synchronization
    block.sync();  // All threads participate in synchronization
}
```

## Practical Exercises

1. **Architecture Analysis**
   - Write a program to query and display GPU properties
   - Compare specifications of different GPU models
   - Analyze the implications for different workloads

2. **Memory Hierarchy Exploration**
   - Implement benchmarks for different memory types
   - Measure bandwidth and latency characteristics
   - Optimize memory access patterns

3. **Parallel Algorithm Design**
   - Implement parallel reduction using different strategies
   - Compare performance of various decomposition approaches
   - Analyze scalability with different problem sizes

## Key Takeaways

- GPUs excel at data-parallel workloads with high arithmetic intensity
- Understanding memory hierarchy is crucial for performance
- Warp-level thinking is essential for efficient GPU programming
- Proper synchronization prevents race conditions and ensures correctness
- Amdahl's Law limits the benefits of parallelization

## Next Steps

After mastering GPU architecture concepts, proceed to:
- CUDA Programming Fundamentals
- Setting up the development environment
- Writing your first CUDA kernels
- Understanding the CUDA memory model
