# Advanced CUDA Programming

*Duration: 2 weeks*

## Overview

This section covers advanced CUDA programming techniques including streams, events, dynamic parallelism, multi-GPU programming, and other sophisticated features that enable high-performance computing applications.

## Learning Objectives

By the end of this section, you will be able to:
- Use CUDA streams for asynchronous execution and overlapping operations
- Implement dynamic parallelism for nested kernel launches
- Develop multi-GPU applications with efficient communication
- Apply persistent thread patterns for specialized workloads
- Optimize applications using advanced CUDA features

## 1. CUDA Streams

### Stream Basics

```cuda
// Basic stream operations
void streamBasics() {
    const int N = 1000000;
    const size_t size = N * sizeof(float);
    
    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Allocate memory
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    cudaMallocHost(&h_a, size);  // Pinned memory for async transfers
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Asynchronous operations
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);
    
    // Launch kernels in different streams
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    vectorAdd<<<gridSize, blockSize, 0, stream1>>>(d_a, d_b, d_c, N);
    
    // Copy result back asynchronously
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream1);
    
    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Stream Synchronization

```cuda
// Advanced stream synchronization
class StreamManager {
private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    
public:
    StreamManager(int num_streams) {
        streams_.resize(num_streams);
        events_.resize(num_streams);
        
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams_[i]);
            cudaEventCreate(&events_[i]);
        }
    }
    
    ~StreamManager() {
        for (size_t i = 0; i < streams_.size(); i++) {
            cudaStreamDestroy(streams_[i]);
            cudaEventDestroy(events_[i]);
        }
    }
    
    cudaStream_t getStream(int index) {
        return streams_[index % streams_.size()];
    }
    
    void recordEvent(int stream_idx) {
        cudaEventRecord(events_[stream_idx], streams_[stream_idx]);
    }
    
    void waitForEvent(int wait_stream_idx, int event_stream_idx) {
        cudaStreamWaitEvent(streams_[wait_stream_idx], events_[event_stream_idx], 0);
    }
    
    void synchronizeAll() {
        for (auto& stream : streams_) {
            cudaStreamSynchronize(stream);
        }
    }
    
    // Pipeline processing example
    void pipelineProcess(float* input, float* output, int total_elements, int chunk_size) {
        int num_chunks = (total_elements + chunk_size - 1) / chunk_size;
        int num_streams = streams_.size();
        
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int stream_idx = chunk % num_streams;
            cudaStream_t stream = streams_[stream_idx];
            
            int offset = chunk * chunk_size;
            int elements = std::min(chunk_size, total_elements - offset);
            
            // Stage 1: Memory transfer H2D
            cudaMemcpyAsync(input + offset, input + offset, 
                           elements * sizeof(float), 
                           cudaMemcpyHostToDevice, stream);
            
            // Stage 2: Kernel execution
            int blockSize = 256;
            int gridSize = (elements + blockSize - 1) / blockSize;
            processChunk<<<gridSize, blockSize, 0, stream>>>(
                input + offset, output + offset, elements);
            
            // Stage 3: Memory transfer D2H
            cudaMemcpyAsync(output + offset, output + offset, 
                           elements * sizeof(float), 
                           cudaMemcpyDeviceToHost, stream);
        }
        
        synchronizeAll();
    }
};

__global__ void processChunk(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrt(input[idx] * input[idx] + 1.0f);
    }
}
```

### Multi-Stream Concurrency

```cuda
// Overlapping computation and communication
void demonstrateOverlap() {
    const int N = 1000000;
    const int num_streams = 4;
    const int chunk_size = N / num_streams;
    
    // Create streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Allocate memory
    float *h_input, *h_output;
    float *d_input, *d_output;
    
    cudaMallocHost(&h_input, N * sizeof(float));
    cudaMallocHost(&h_output, N * sizeof(float));
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 0.1f;
    }
    
    // Process in overlapping chunks
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        int size = (i == num_streams - 1) ? N - offset : chunk_size;
        
        // Async memory copy H2D
        cudaMemcpyAsync(d_input + offset, h_input + offset, 
                       size * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[i]);
        
        // Kernel launch
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        complexProcessing<<<gridSize, blockSize, 0, streams[i]>>>(
            d_input + offset, d_output + offset, size);
        
        // Async memory copy D2H
        cudaMemcpyAsync(h_output + offset, d_output + offset, 
                       size * sizeof(float), 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = processValue(h_input[i]);
        if (abs(h_output[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_input); cudaFreeHost(h_output);
    cudaFree(d_input); cudaFree(d_output);
}

__global__ void complexProcessing(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Complex computation
        output[idx] = sinf(x) * cosf(x) + expf(-x * x) * logf(x + 1.0f);
    }
}

float processValue(float x) {
    return sinf(x) * cosf(x) + expf(-x * x) * logf(x + 1.0f);
}
```

## 2. CUDA Events

### Event-Based Timing and Synchronization

```cuda
// Advanced event usage
class EventManager {
private:
    std::vector<cudaEvent_t> events_;
    std::map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> timers_;
    
public:
    EventManager(int num_events) {
        events_.resize(num_events);
        for (int i = 0; i < num_events; i++) {
            cudaEventCreate(&events_[i]);
        }
    }
    
    ~EventManager() {
        for (auto& event : events_) {
            cudaEventDestroy(event);
        }
        for (auto& timer : timers_) {
            cudaEventDestroy(timer.second.first);
            cudaEventDestroy(timer.second.second);
        }
    }
    
    void startTimer(const std::string& name) {
        if (timers_.find(name) == timers_.end()) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            timers_[name] = std::make_pair(start, stop);
        }
        cudaEventRecord(timers_[name].first);
    }
    
    float stopTimer(const std::string& name) {
        if (timers_.find(name) == timers_.end()) {
            return -1.0f;
        }
        
        cudaEventRecord(timers_[name].second);
        cudaEventSynchronize(timers_[name].second);
        
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, timers_[name].first, timers_[name].second);
        
        return elapsed_time;
    }
    
    // Callback function example
    static void CUDART_CB eventCallback(cudaStream_t stream, cudaError_t status, void* data) {
        std::string* message = static_cast<std::string*>(data);
        std::cout << "Event callback: " << *message << std::endl;
        delete message;
    }
    
    void recordCallback(cudaStream_t stream, const std::string& message) {
        std::string* msg = new std::string(message);
        cudaStreamAddCallback(stream, eventCallback, msg, 0);
    }
};

// Performance profiling with events
void profileKernelExecution() {
    EventManager event_mgr(10);
    
    const int N = 1000000;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Initialize data
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    initializeKernel<<<1000, 256, 0, stream>>>(d_data, N);
    
    // Profile different kernels
    std::vector<std::string> kernel_names = {
        "Simple", "Optimized", "Shared Memory", "Unrolled"
    };
    
    for (const auto& name : kernel_names) {
        event_mgr.startTimer(name);
        
        if (name == "Simple") {
            simpleKernel<<<1000, 256, 0, stream>>>(d_data, N);
        } else if (name == "Optimized") {
            optimizedKernel<<<1000, 256, 0, stream>>>(d_data, N);
        } else if (name == "Shared Memory") {
            sharedMemoryKernel<<<1000, 256, 256 * sizeof(float), stream>>>(d_data, N);
        } else if (name == "Unrolled") {
            unrolledKernel<<<1000, 256, 0, stream>>>(d_data, N);
        }
        
        float time = event_mgr.stopTimer(name);
        std::cout << name << " kernel: " << time << " ms" << std::endl;
    }
    
    cudaFree(d_data);
    cudaStreamDestroy(stream);
}

__global__ void initializeKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 0.1f;
    }
}

__global__ void simpleKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}

__global__ void optimizedKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = sqrtf(fmaf(x, x, 1.0f));  // Use fused multiply-add
    }
}

__global__ void sharedMemoryKernel(float* data, int n) {
    extern __shared__ float shared_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        shared_data[tid] = data[idx];
    }
    
    __syncthreads();
    
    if (idx < n) {
        float x = shared_data[tid];
        shared_data[tid] = sqrtf(x * x + 1.0f);
    }
    
    __syncthreads();
    
    if (idx < n) {
        data[idx] = shared_data[tid];
    }
}

__global__ void unrolledKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread
    for (int i = 0; i < 4 && idx * 4 + i < n; i++) {
        int index = idx * 4 + i;
        float x = data[index];
        data[index] = sqrtf(x * x + 1.0f);
    }
}
```

## 3. Dynamic Parallelism

### Nested Kernel Launches

```cuda
// Dynamic parallelism example
__global__ void parentKernel(float* data, int n, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Process current level
        data[idx] = sqrtf(data[idx]);
        
        // Launch child kernel if more depth needed
        if (depth > 0 && idx == 0) {  // Only first thread launches child
            int child_blocks = (n + 255) / 256;
            childKernel<<<child_blocks, 256>>>(data, n, depth - 1);
            cudaDeviceSynchronize();  // Wait for child to complete
        }
    }
}

__global__ void childKernel(float* data, int n, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Child processing
        data[idx] = data[idx] * data[idx] + 1.0f;
        
        // Recursive launch if needed
        if (depth > 0 && threadIdx.x == 0 && blockIdx.x == 0) {
            int grandchild_blocks = (n + 255) / 256;
            childKernel<<<grandchild_blocks, 256>>>(data, n, depth - 1);
        }
    }
}

// Adaptive mesh refinement using dynamic parallelism
struct MeshCell {
    float value;
    float error;
    int level;
    bool needs_refinement;
};

__global__ void adaptiveMeshKernel(MeshCell* cells, int num_cells, int max_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_cells) return;
    
    MeshCell& cell = cells[idx];
    
    // Calculate error metric
    calculateError(cell);
    
    // Check if refinement is needed
    if (cell.error > 0.1f && cell.level < max_level) {
        cell.needs_refinement = true;
        
        // Launch refinement kernel
        if (threadIdx.x == 0) {  // One thread per block launches refinement
            refineCell<<<1, 4>>>(cells, idx, num_cells);
        }
    }
}

__device__ void calculateError(MeshCell& cell) {
    // Simple error calculation
    cell.error = abs(cell.value - 0.5f);
}

__global__ void refineCell(MeshCell* cells, int parent_idx, int num_cells) {
    int child_idx = threadIdx.x;  // 4 children per parent
    
    if (parent_idx * 4 + child_idx < num_cells) {
        MeshCell& parent = cells[parent_idx];
        MeshCell& child = cells[parent_idx * 4 + child_idx];
        
        // Initialize child cell
        child.value = parent.value + (child_idx - 2) * 0.1f;
        child.level = parent.level + 1;
        child.needs_refinement = false;
        
        // Recursively refine if needed
        calculateError(child);
        if (child.error > 0.05f && child.level < 5) {
            refineCell<<<1, 4>>>(cells, parent_idx * 4 + child_idx, num_cells);
        }
    }
}
```

### Work Queue Pattern

```cuda
// Work queue implementation with dynamic parallelism
struct WorkItem {
    int start_index;
    int end_index;
    int processing_level;
    float* data;
};

__device__ __managed__ WorkItem* work_queue;
__device__ __managed__ int queue_size;
__device__ __managed__ int queue_capacity;

__global__ void workQueueProcessor(WorkItem* queue, int* size, int capacity) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (true) {
        // Try to get work item
        int work_idx = atomicAdd(size, -1) - 1;
        
        if (work_idx < 0) {
            // No more work
            break;
        }
        
        WorkItem item = queue[work_idx];
        
        // Process work item
        processWorkItem(item);
        
        // Generate new work if needed
        if (shouldSubdivide(item)) {
            WorkItem new_items[2];
            subdivideWork(item, new_items);
            
            // Add new work to queue
            for (int i = 0; i < 2; i++) {
                int new_idx = atomicAdd(size, 1);
                if (new_idx < capacity) {
                    queue[new_idx] = new_items[i];
                }
            }
        }
    }
}

__device__ void processWorkItem(const WorkItem& item) {
    for (int i = item.start_index; i < item.end_index; i++) {
        // Complex processing based on level
        float value = item.data[i];
        for (int level = 0; level < item.processing_level; level++) {
            value = sinf(value) + cosf(value);
        }
        item.data[i] = value;
    }
}

__device__ bool shouldSubdivide(const WorkItem& item) {
    int work_size = item.end_index - item.start_index;
    return work_size > 1000 && item.processing_level < 3;
}

__device__ void subdivideWork(const WorkItem& item, WorkItem* new_items) {
    int mid = (item.start_index + item.end_index) / 2;
    
    new_items[0] = {item.start_index, mid, item.processing_level + 1, item.data};
    new_items[1] = {mid, item.end_index, item.processing_level + 1, item.data};
}
```

## 4. Multi-GPU Programming

### Device Management

```cuda
// Multi-GPU device management
class MultiGPUManager {
private:
    int num_devices_;
    std::vector<cudaDeviceProp> device_props_;
    std::vector<cudaStream_t> streams_;
    
public:
    MultiGPUManager() {
        cudaGetDeviceCount(&num_devices_);
        
        device_props_.resize(num_devices_);
        streams_.resize(num_devices_);
        
        for (int i = 0; i < num_devices_; i++) {
            cudaSetDevice(i);
            cudaGetDeviceProperties(&device_props_[i], i);
            cudaStreamCreate(&streams_[i]);
            
            std::cout << "Device " << i << ": " << device_props_[i].name 
                      << " (Memory: " << device_props_[i].totalGlobalMem / (1024*1024) 
                      << " MB)" << std::endl;
        }
        
        // Check P2P capabilities
        checkP2PCapabilities();
    }
    
    ~MultiGPUManager() {
        for (int i = 0; i < num_devices_; i++) {
            cudaSetDevice(i);
            cudaStreamDestroy(streams_[i]);
        }
    }
    
    void checkP2PCapabilities() {
        std::cout << "\nP2P Access Matrix:" << std::endl;
        for (int i = 0; i < num_devices_; i++) {
            for (int j = 0; j < num_devices_; j++) {
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                std::cout << can_access << " ";
            }
            std::cout << std::endl;
        }
        
        // Enable P2P access
        for (int i = 0; i < num_devices_; i++) {
            cudaSetDevice(i);
            for (int j = 0; j < num_devices_; j++) {
                if (i != j) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, i, j);
                    if (can_access) {
                        cudaDeviceEnablePeerAccess(j, 0);
                    }
                }
            }
        }
    }
    
    // Distribute work across GPUs
    void distributeVectorAddition(float* h_a, float* h_b, float* h_c, int total_size) {
        int elements_per_gpu = total_size / num_devices_;
        
        std::vector<float*> d_a(num_devices_);
        std::vector<float*> d_b(num_devices_);
        std::vector<float*> d_c(num_devices_);
        
        // Allocate memory on each GPU
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);
            
            int gpu_elements = (gpu == num_devices_ - 1) ? 
                              total_size - gpu * elements_per_gpu : elements_per_gpu;
            
            cudaMalloc(&d_a[gpu], gpu_elements * sizeof(float));
            cudaMalloc(&d_b[gpu], gpu_elements * sizeof(float));
            cudaMalloc(&d_c[gpu], gpu_elements * sizeof(float));
            
            // Copy data to GPU
            int offset = gpu * elements_per_gpu;
            cudaMemcpyAsync(d_a[gpu], h_a + offset, gpu_elements * sizeof(float),
                           cudaMemcpyHostToDevice, streams_[gpu]);
            cudaMemcpyAsync(d_b[gpu], h_b + offset, gpu_elements * sizeof(float),
                           cudaMemcpyHostToDevice, streams_[gpu]);
        }
        
        // Launch kernels on each GPU
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);
            
            int gpu_elements = (gpu == num_devices_ - 1) ? 
                              total_size - gpu * elements_per_gpu : elements_per_gpu;
            
            int blockSize = 256;
            int gridSize = (gpu_elements + blockSize - 1) / blockSize;
            
            vectorAdd<<<gridSize, blockSize, 0, streams_[gpu]>>>(
                d_a[gpu], d_b[gpu], d_c[gpu], gpu_elements);
        }
        
        // Copy results back
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);
            
            int gpu_elements = (gpu == num_devices_ - 1) ? 
                              total_size - gpu * elements_per_gpu : elements_per_gpu;
            int offset = gpu * elements_per_gpu;
            
            cudaMemcpyAsync(h_c + offset, d_c[gpu], gpu_elements * sizeof(float),
                           cudaMemcpyDeviceToHost, streams_[gpu]);
        }
        
        // Synchronize all GPUs
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);
            cudaStreamSynchronize(streams_[gpu]);
        }
        
        // Cleanup
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);
            cudaFree(d_a[gpu]);
            cudaFree(d_b[gpu]);
            cudaFree(d_c[gpu]);
        }
    }
    
    int getDeviceCount() const { return num_devices_; }
    cudaStream_t getStream(int device) const { return streams_[device]; }
};
```

### P2P Communication

```cuda
// Peer-to-peer communication example
void demonstrateP2PCommunication() {
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    
    if (num_devices < 2) {
        std::cout << "Need at least 2 GPUs for P2P demo" << std::endl;
        return;
    }
    
    const int N = 1000000;
    const size_t size = N * sizeof(float);
    
    // Allocate memory on GPU 0
    cudaSetDevice(0);
    float* d_data0;
    cudaMalloc(&d_data0, size);
    
    // Initialize data on GPU 0
    initializeData<<<1000, 256>>>(d_data0, N);
    cudaDeviceSynchronize();
    
    // Allocate memory on GPU 1
    cudaSetDevice(1);
    float* d_data1;
    cudaMalloc(&d_data1, size);
    
    // Enable P2P access
    int can_access_01, can_access_10;
    cudaDeviceCanAccessPeer(&can_access_01, 0, 1);
    cudaDeviceCanAccessPeer(&can_access_10, 1, 0);
    
    if (can_access_01 && can_access_10) {
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        
        // Direct P2P copy
        cudaMemcpyPeer(d_data1, 1, d_data0, 0, size);
        
        // Process on GPU 1
        cudaSetDevice(1);
        processData<<<1000, 256>>>(d_data1, N);
        
        // Copy result back to GPU 0
        cudaMemcpyPeer(d_data0, 0, d_data1, 1, size);
        
        std::cout << "P2P communication successful" << std::endl;
    } else {
        std::cout << "P2P not supported between devices 0 and 1" << std::endl;
        
        // Use staging through host memory
        cudaSetDevice(0);
        float* h_staging;
        cudaMallocHost(&h_staging, size);
        
        cudaMemcpy(h_staging, d_data0, size, cudaMemcpyDeviceToHost);
        
        cudaSetDevice(1);
        cudaMemcpy(d_data1, h_staging, size, cudaMemcpyHostToDevice);
        
        processData<<<1000, 256>>>(d_data1, N);
        
        cudaMemcpy(h_staging, d_data1, size, cudaMemcpyDeviceToHost);
        
        cudaSetDevice(0);
        cudaMemcpy(d_data0, h_staging, size, cudaMemcpyHostToDevice);
        
        cudaFreeHost(h_staging);
    }
    
    // Cleanup
    cudaSetDevice(0);
    cudaFree(d_data0);
    cudaSetDevice(1);
    cudaFree(d_data1);
}

__global__ void initializeData(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 0.1f;
    }
}

__global__ void processData(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}
```

## 5. Persistent Threads

### Long-Running Kernels

```cuda
// Persistent thread pattern for continuous processing
__global__ void persistentThreadKernel(float* input_queue, float* output_queue,
                                      int* queue_head, int* queue_tail,
                                      bool* terminate_flag, int queue_capacity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    
    while (!(*terminate_flag)) {
        // Try to get work from queue
        int work_index = atomicAdd(queue_head, 1);
        
        if (work_index >= *queue_tail) {
            // No work available, wait a bit
            __nanosleep(1000);  // 1 microsecond
            continue;
        }
        
        // Process work item
        int queue_idx = work_index % queue_capacity;
        float input_value = input_queue[queue_idx];
        
        // Complex processing
        float result = 0.0f;
        for (int i = 0; i < 100; i++) {
            result += sinf(input_value + i * 0.01f) * cosf(input_value + i * 0.02f);
        }
        
        // Store result
        output_queue[queue_idx] = result;
        
        // Indicate completion
        __threadfence();
    }
}

// Producer-consumer pattern
class PersistentKernelManager {
private:
    float* d_input_queue;
    float* d_output_queue;
    int* d_queue_head;
    int* d_queue_tail;
    bool* d_terminate_flag;
    int queue_capacity;
    
    cudaStream_t kernel_stream;
    
public:
    PersistentKernelManager(int capacity) : queue_capacity(capacity) {
        // Allocate unified memory for queues
        cudaMallocManaged(&d_input_queue, capacity * sizeof(float));
        cudaMallocManaged(&d_output_queue, capacity * sizeof(float));
        cudaMallocManaged(&d_queue_head, sizeof(int));
        cudaMallocManaged(&d_queue_tail, sizeof(int));
        cudaMallocManaged(&d_terminate_flag, sizeof(bool));
        
        // Initialize
        *d_queue_head = 0;
        *d_queue_tail = 0;
        *d_terminate_flag = false;
        
        cudaStreamCreate(&kernel_stream);
        
        // Launch persistent kernel
        persistentThreadKernel<<<32, 256, 0, kernel_stream>>>(
            d_input_queue, d_output_queue, d_queue_head, d_queue_tail,
            d_terminate_flag, queue_capacity);
    }
    
    ~PersistentKernelManager() {
        // Signal termination
        *d_terminate_flag = true;
        cudaStreamSynchronize(kernel_stream);
        
        // Cleanup
        cudaFree(d_input_queue);
        cudaFree(d_output_queue);
        cudaFree(d_queue_head);
        cudaFree(d_queue_tail);
        cudaFree(d_terminate_flag);
        cudaStreamDestroy(kernel_stream);
    }
    
    void addWork(const std::vector<float>& work_items) {
        for (float item : work_items) {
            int tail_idx = atomicAdd(d_queue_tail, 1) % queue_capacity;
            d_input_queue[tail_idx] = item;
        }
    }
    
    std::vector<float> getResults(int num_results) {
        std::vector<float> results;
        
        while (results.size() < num_results) {
            if (*d_queue_head > results.size()) {
                int result_idx = results.size() % queue_capacity;
                results.push_back(d_output_queue[result_idx]);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        return results;
    }
};
```

### Task Stealing

```cuda
// Task stealing implementation
struct TaskQueue {
    float* tasks;
    int* head;
    int* tail;
    int capacity;
};

__global__ void taskStealingKernel(TaskQueue* queues, int num_queues,
                                  float* results, bool* terminate_flag) {
    int worker_id = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each block is a worker with its own queue
    TaskQueue& my_queue = queues[worker_id];
    
    while (!(*terminate_flag)) {
        bool found_work = false;
        
        // Try to get work from own queue first
        if (tid == 0) {  // Only one thread per block manages queue
            if (my_queue.head[0] < my_queue.tail[0]) {
                int task_idx = atomicAdd(my_queue.head, 1);
                if (task_idx < my_queue.tail[0]) {
                    found_work = true;
                    // Process task
                    float task_data = my_queue.tasks[task_idx % my_queue.capacity];
                    results[task_idx] = processTask(task_data);
                }
            }
        }
        
        __syncthreads();
        
        // If no local work, try to steal from other queues
        if (!found_work && tid == 0) {
            for (int victim = 0; victim < num_queues; victim++) {
                if (victim == worker_id) continue;
                
                TaskQueue& victim_queue = queues[victim];
                
                // Try to steal half of victim's remaining work
                int victim_remaining = victim_queue.tail[0] - victim_queue.head[0];
                if (victim_remaining > 1) {
                    int steal_count = victim_remaining / 2;
                    int steal_start = atomicAdd(victim_queue.head, steal_count);
                    
                    // Move stolen tasks to own queue
                    for (int i = 0; i < steal_count; i++) {
                        int victim_idx = (steal_start + i) % victim_queue.capacity;
                        int my_idx = atomicAdd(my_queue.tail, 1) % my_queue.capacity;
                        my_queue.tasks[my_idx] = victim_queue.tasks[victim_idx];
                    }
                    break;
                }
            }
        }
        
        __syncthreads();
        
        // If still no work, wait a bit
        if (!found_work) {
            __nanosleep(1000);
        }
    }
}

__device__ float processTask(float task_data) {
    // Simulate complex computation
    float result = task_data;
    for (int i = 0; i < 50; i++) {
        result = sinf(result) + cosf(result * 0.1f);
    }
    return result;
}
```

## Practical Exercises

1. **Stream Optimization**
   - Implement overlapped computation and communication
   - Compare performance with and without streams
   - Optimize pipeline processing for continuous data

2. **Dynamic Parallelism Applications**
   - Implement recursive algorithms (quicksort, tree traversal)
   - Create adaptive algorithms that adjust parallelism at runtime
   - Compare with static parallelism approaches

3. **Multi-GPU Scaling**
   - Implement matrix multiplication across multiple GPUs
   - Measure scaling efficiency with different workloads
   - Optimize P2P communication patterns

4. **Persistent Thread Patterns**
   - Create a real-time processing system
   - Implement work-stealing scheduler
   - Compare with traditional kernel launches

## Key Takeaways

- Streams enable asynchronous execution and better resource utilization
- Dynamic parallelism allows for adaptive algorithms but has overhead
- Multi-GPU programming requires careful load balancing and communication
- Persistent threads are useful for continuous processing workloads
- Advanced features should be used judiciously based on application needs

## Next Steps

After mastering advanced CUDA programming, proceed to:
- CUDA Libraries and Tools
- Domain-specific optimizations
- Integration with other parallel computing frameworks
- Performance analysis and debugging tools
