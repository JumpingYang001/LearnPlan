# Memory Management Techniques

*Duration: 1 week*

## Overview

Advanced memory management is crucial for achieving optimal CUDA performance. This section covers unified memory advanced features, texture memory, constant memory, and memory pool allocation techniques.

## Unified Memory Advanced Features

### Memory Advising

```cpp
#include <cuda_runtime.h>

// Memory advising for performance optimization
class UnifiedMemoryManager {
private:
    void* data;
    size_t size;
    int device_id;
    
public:
    UnifiedMemoryManager(size_t bytes, int device = 0) : size(bytes), device_id(device) {
        // Allocate unified memory
        cudaMallocManaged(&data, size);
        
        // Set memory advice for optimal performance
        optimize_memory_placement();
    }
    
    ~UnifiedMemoryManager() {
        cudaFree(data);
    }
    
    void optimize_memory_placement() {
        // Advise that data will be mostly read
        cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device_id);
        
        // Set preferred location
        cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, device_id);
        
        // Advise that data will be accessed by device
        cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, device_id);
        
        // Advise that data will be accessed by CPU
        cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    }
    
    void prefetch_to_device() {
        cudaMemPrefetchAsync(data, size, device_id);
    }
    
    void prefetch_to_host() {
        cudaMemPrefetchAsync(data, size, cudaCpuDeviceId);
    }
    
    template<typename T>
    T* get_ptr() { return static_cast<T*>(data); }
};

// Example usage with different access patterns
void unified_memory_example() {
    const size_t n = 1000000;
    UnifiedMemoryManager um_manager(n * sizeof(float));
    
    float* data = um_manager.get_ptr<float>();
    
    // Initialize on CPU
    for (size_t i = 0; i < n; i++) {
        data[i] = static_cast<float>(i);
    }
    
    // Prefetch to GPU before kernel execution
    um_manager.prefetch_to_device();
    cudaDeviceSynchronize();
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    process_kernel<<<grid, block>>>(data, n);
    cudaDeviceSynchronize();
    
    // Prefetch back to CPU for result processing
    um_manager.prefetch_to_host();
    cudaDeviceSynchronize();
    
    // Process results on CPU
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    
    printf("Sum: %f\n", sum);
}

__global__ void process_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx] * 2.0f);
    }
}
```

### Memory Migration and Hints

```cpp
// Advanced unified memory management with migration hints
class AdaptiveMemoryManager {
private:
    struct MemoryRegion {
        void* ptr;
        size_t size;
        cudaMemoryType current_location;
        std::vector<int> accessing_devices;
        size_t access_count;
        std::chrono::high_resolution_clock::time_point last_access;
    };
    
    std::vector<MemoryRegion> regions;
    
public:
    void* allocate_adaptive(size_t size, const std::vector<int>& devices) {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        
        MemoryRegion region;
        region.ptr = ptr;
        region.size = size;
        region.current_location = cudaMemoryTypeUnregistered;
        region.accessing_devices = devices;
        region.access_count = 0;
        region.last_access = std::chrono::high_resolution_clock::now();
        
        regions.push_back(region);
        
        // Set initial placement hints
        if (!devices.empty()) {
            cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, devices[0]);
            
            for (int device : devices) {
                cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);
            }
        }
        
        return ptr;
    }
    
    void access_pattern_hint(void* ptr, int device, bool read_heavy = true) {
        auto it = std::find_if(regions.begin(), regions.end(),
                             [ptr](const MemoryRegion& r) { return r.ptr == ptr; });
        
        if (it != regions.end()) {
            it->access_count++;
            it->last_access = std::chrono::high_resolution_clock::now();
            
            if (read_heavy) {
                cudaMemAdvise(ptr, it->size, cudaMemAdviseSetReadMostly, device);
            }
            
            // Migrate if access pattern changes
            migrate_if_beneficial(it, device);
        }
    }
    
private:
    void migrate_if_beneficial(std::vector<MemoryRegion>::iterator region, int target_device) {
        // Simple heuristic: migrate if frequently accessed from target device
        if (region->access_count > 10 && 
            std::find(region->accessing_devices.begin(), region->accessing_devices.end(), target_device) 
            != region->accessing_devices.end()) {
            
            cudaMemPrefetchAsync(region->ptr, region->size, target_device);
        }
    }
};

// Asynchronous prefetching with streams
void async_prefetch_example() {
    const size_t chunk_size = 1000000 * sizeof(float);
    const int num_chunks = 4;
    
    std::vector<float*> chunks(num_chunks);
    std::vector<cudaStream_t> streams(num_chunks);
    
    // Allocate chunks and create streams
    for (int i = 0; i < num_chunks; i++) {
        cudaMallocManaged(&chunks[i], chunk_size);
        cudaStreamCreate(&streams[i]);
        
        // Initialize data on CPU
        for (size_t j = 0; j < chunk_size / sizeof(float); j++) {
            chunks[i][j] = static_cast<float>(i * 1000 + j);
        }
    }
    
    // Pipeline: prefetch, compute, prefetch next
    for (int i = 0; i < num_chunks; i++) {
        // Prefetch current chunk
        cudaMemPrefetchAsync(chunks[i], chunk_size, 0, streams[i]);
        
        // Prefetch next chunk (if exists)
        if (i + 1 < num_chunks) {
            cudaMemPrefetchAsync(chunks[i + 1], chunk_size, 0, streams[(i + 1) % num_chunks]);
        }
        
        // Launch computation on current chunk
        dim3 block(256);
        dim3 grid((chunk_size / sizeof(float) + block.x - 1) / block.x);
        
        compute_kernel<<<grid, block, 0, streams[i]>>>(chunks[i], chunk_size / sizeof(float));
    }
    
    // Synchronize all streams
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(chunks[i]);
    }
}

__global__ void compute_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sinf(data[idx]) * cosf(data[idx]);
    }
}
```

## Texture Memory

### Texture Objects and References

```cpp
// Texture memory for spatial locality and filtering
class TextureMemoryManager {
private:
    cudaArray_t cuda_array;
    cudaTextureObject_t texture_obj;
    cudaResourceDesc res_desc;
    cudaTextureDesc tex_desc;
    
    int width, height;
    
public:
    TextureMemoryManager(int w, int h) : width(w), height(h) {
        // Allocate CUDA array
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaMallocArray(&cuda_array, &channel_desc, width, height);
        
        // Setup resource descriptor
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;
        
        // Setup texture descriptor
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1; // Use normalized coordinates [0,1]
        
        // Create texture object
        cudaCreateTextureObject(&texture_obj, &res_desc, &tex_desc, nullptr);
    }
    
    ~TextureMemoryManager() {
        cudaDestroyTextureObject(texture_obj);
        cudaFreeArray(cuda_array);
    }
    
    void upload_data(float* host_data) {
        cudaMemcpy2DToArray(cuda_array, 0, 0, host_data, 
                           width * sizeof(float), width * sizeof(float), height,
                           cudaMemcpyHostToDevice);
    }
    
    cudaTextureObject_t get_texture() const { return texture_obj; }
    
    // Bilinear interpolation example
    void apply_bilinear_filter(float* output) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        bilinear_kernel<<<grid, block>>>(texture_obj, output, width, height);
    }
    
    // Image scaling using texture interpolation
    void scale_image(float* output, int new_width, int new_height) {
        dim3 block(16, 16);
        dim3 grid((new_width + block.x - 1) / block.x, (new_height + block.y - 1) / block.y);
        
        scale_kernel<<<grid, block>>>(texture_obj, output, new_width, new_height, 
                                     (float)width / new_width, (float)height / new_height);
    }
};

// Kernels using texture memory
__global__ void bilinear_kernel(cudaTextureObject_t tex, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Normalized coordinates
    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;
    
    // Sample texture with bilinear interpolation
    float value = tex2D<float>(tex, u, v);
    
    output[y * width + x] = value;
}

__global__ void scale_kernel(cudaTextureObject_t tex, float* output, 
                           int new_width, int new_height, float scale_x, float scale_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= new_width || y >= new_height) return;
    
    // Map to original coordinates
    float u = (x + 0.5f) * scale_x / new_width;
    float v = (y + 0.5f) * scale_y / new_height;
    
    // Sample with automatic interpolation
    float value = tex2D<float>(tex, u, v);
    
    output[y * new_width + x] = value;
}

// 3D texture example
class Texture3D {
private:
    cudaArray_t cuda_array_3d;
    cudaTextureObject_t texture_obj_3d;
    
public:
    Texture3D(int width, int height, int depth) {
        // Allocate 3D CUDA array
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(width, height, depth);
        
        cudaMalloc3DArray(&cuda_array_3d, &channel_desc, extent);
        
        // Setup resource and texture descriptors
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array_3d;
        
        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.addressMode[2] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;
        
        cudaCreateTextureObject(&texture_obj_3d, &res_desc, &tex_desc, nullptr);
    }
    
    void upload_volume_data(float* host_data, int width, int height, int depth) {
        cudaMemcpy3DParms copy_params = {0};
        copy_params.srcPtr = make_cudaPitchedPtr(host_data, width * sizeof(float), width, height);
        copy_params.dstArray = cuda_array_3d;
        copy_params.extent = make_cudaExtent(width, height, depth);
        copy_params.kind = cudaMemcpyHostToDevice;
        
        cudaMemcpy3D(&copy_params);
    }
    
    cudaTextureObject_t get_texture() const { return texture_obj_3d; }
};

// Volume rendering kernel
__global__ void volume_render_kernel(cudaTextureObject_t volume_tex, float* output,
                                   int image_width, int image_height,
                                   float3 eye_pos, float3 view_dir, float3 up_dir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= image_width || y >= image_height) return;
    
    // Calculate ray direction
    float u = (x - image_width * 0.5f) / image_width;
    float v = (y - image_height * 0.5f) / image_height;
    
    float3 right = normalize(cross(view_dir, up_dir));
    float3 ray_dir = normalize(view_dir + u * right + v * up_dir);
    
    // Ray marching through volume
    float accumulated = 0.0f;
    float step_size = 0.01f;
    int max_steps = 100;
    
    for (int step = 0; step < max_steps; step++) {
        float3 sample_pos = eye_pos + step * step_size * ray_dir;
        
        // Sample volume texture
        float density = tex3D<float>(volume_tex, 
                                   sample_pos.x, sample_pos.y, sample_pos.z);
        
        accumulated += density * step_size;
        
        if (accumulated > 1.0f) break;
    }
    
    output[y * image_width + x] = accumulated;
}
```

## Constant Memory

### Broadcast Patterns and Caching

```cpp
// Constant memory for read-only data
__constant__ float filter_kernel[25]; // 5x5 filter kernel
__constant__ float transform_matrix[16]; // 4x4 transformation matrix
__constant__ float lookup_table[1024]; // Lookup table

// Convolution using constant memory
__global__ void convolution_constant_kernel(float* input, float* output,
                                          int width, int height, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int half_filter = filter_size / 2;
    
    for (int fy = -half_filter; fy <= half_filter; fy++) {
        for (int fx = -half_filter; fx <= half_filter; fx++) {
            int nx = min(max(x + fx, 0), width - 1);
            int ny = min(max(y + fy, 0), height - 1);
            
            int filter_idx = (fy + half_filter) * filter_size + (fx + half_filter);
            sum += input[ny * width + nx] * filter_kernel[filter_idx];
        }
    }
    
    output[y * width + x] = sum;
}

// Matrix transformation using constant memory
__global__ void transform_points_kernel(float4* points, float4* transformed_points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float4 p = points[idx];
    
    // Matrix multiplication using constant memory
    transformed_points[idx] = make_float4(
        transform_matrix[0] * p.x + transform_matrix[1] * p.y + transform_matrix[2] * p.z + transform_matrix[3] * p.w,
        transform_matrix[4] * p.x + transform_matrix[5] * p.y + transform_matrix[6] * p.z + transform_matrix[7] * p.w,
        transform_matrix[8] * p.x + transform_matrix[9] * p.y + transform_matrix[10] * p.z + transform_matrix[11] * p.w,
        transform_matrix[12] * p.x + transform_matrix[13] * p.y + transform_matrix[14] * p.z + transform_matrix[15] * p.w
    );
}

// Constant memory management class
class ConstantMemoryManager {
public:
    template<typename T>
    static void upload_to_constant(const T* host_data, T* constant_symbol, size_t count) {
        cudaMemcpyToSymbol(constant_symbol, host_data, count * sizeof(T));
    }
    
    static void set_filter_kernel(const float* kernel, int size) {
        if (size <= 25) {
            cudaMemcpyToSymbol(filter_kernel, kernel, size * sizeof(float));
        }
    }
    
    static void set_transform_matrix(const float* matrix) {
        cudaMemcpyToSymbol(transform_matrix, matrix, 16 * sizeof(float));
    }
    
    static void set_lookup_table(const float* table, int size) {
        if (size <= 1024) {
            cudaMemcpyToSymbol(lookup_table, table, size * sizeof(float));
        }
    }
};

// Example usage
void constant_memory_example() {
    // Setup 5x5 Gaussian filter
    float gaussian_kernel[25];
    float sigma = 1.0f;
    int filter_size = 5;
    
    // Generate Gaussian kernel
    for (int y = 0; y < filter_size; y++) {
        for (int x = 0; x < filter_size; x++) {
            int dx = x - filter_size/2;
            int dy = y - filter_size/2;
            gaussian_kernel[y * filter_size + x] = 
                expf(-(dx*dx + dy*dy) / (2 * sigma * sigma));
        }
    }
    
    // Upload to constant memory
    ConstantMemoryManager::set_filter_kernel(gaussian_kernel, 25);
    
    // Setup transformation matrix (rotation)
    float angle = M_PI / 4; // 45 degrees
    float rotation_matrix[16] = {
        cosf(angle), -sinf(angle), 0, 0,
        sinf(angle),  cosf(angle), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    
    ConstantMemoryManager::set_transform_matrix(rotation_matrix);
    
    // Now kernels can use the constant memory efficiently
}
```

## Memory Pool Allocation

### CUDA Memory Pools

```cpp
// Modern CUDA memory pool (CUDA 11.2+)
class CudaMemoryPool {
private:
    cudaMemPool_t memory_pool;
    int device_id;
    size_t pool_size;
    
public:
    CudaMemoryPool(int device = 0, size_t initial_size = 1024 * 1024 * 1024) // 1GB
        : device_id(device), pool_size(initial_size) {
        
        cudaSetDevice(device_id);
        
        // Get default memory pool
        cudaDeviceGetDefaultMemPool(&memory_pool, device_id);
        
        // Set pool properties
        cudaMemPoolProps pool_props = {};
        pool_props.allocType = cudaMemAllocationTypePinned;
        pool_props.handleTypes = cudaMemHandleTypeNone;
        pool_props.location.type = cudaMemLocationTypeDevice;
        pool_props.location.id = device_id;
        
        // Create custom memory pool
        cudaMemPoolCreate(&memory_pool, &pool_props);
        
        // Set access for current device
        cudaMemAccessDesc access_desc = {};
        access_desc.location.type = cudaMemLocationTypeDevice;
        access_desc.location.id = device_id;
        access_desc.flags = cudaMemAccessFlagsProtReadWrite;
        
        cudaMemPoolSetAccess(memory_pool, &access_desc, 1);
        
        // Pre-allocate pool
        uint64_t setThreshold = pool_size;
        cudaMemPoolSetAttribute(memory_pool, cudaMemPoolAttrReleaseThreshold, &setThreshold);
    }
    
    ~CudaMemoryPool() {
        cudaMemPoolDestroy(memory_pool);
    }
    
    void* allocate(size_t size, cudaStream_t stream = 0) {
        void* ptr;
        cudaMallocFromPoolAsync(&ptr, size, memory_pool, stream);
        return ptr;
    }
    
    void deallocate(void* ptr, cudaStream_t stream = 0) {
        cudaFreeAsync(ptr, stream);
    }
    
    void trim() {
        // Trim unused memory from pool
        cudaMemPoolTrimTo(memory_pool, 0);
    }
    
    size_t get_used_memory() {
        uint64_t used_bytes;
        cudaMemPoolGetAttribute(memory_pool, cudaMemPoolAttrUsedMemCurrent, &used_bytes);
        return static_cast<size_t>(used_bytes);
    }
    
    size_t get_reserved_memory() {
        uint64_t reserved_bytes;
        cudaMemPoolGetAttribute(memory_pool, cudaMemPoolAttrReservedMemCurrent, &reserved_bytes);
        return static_cast<size_t>(reserved_bytes);
    }
};

// Custom allocator using memory pools
template<typename T>
class PoolAllocator {
private:
    CudaMemoryPool* pool;
    cudaStream_t stream;
    
public:
    PoolAllocator(CudaMemoryPool* memory_pool, cudaStream_t allocation_stream = 0)
        : pool(memory_pool), stream(allocation_stream) {}
    
    T* allocate(size_t count) {
        return static_cast<T*>(pool->allocate(count * sizeof(T), stream));
    }
    
    void deallocate(T* ptr) {
        pool->deallocate(ptr, stream);
    }
};

// Stream-ordered allocation example
void stream_ordered_allocation_example() {
    CudaMemoryPool pool;
    
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const size_t array_size = 1000000;
    std::vector<float*> arrays(num_streams);
    
    // Allocate arrays on different streams
    for (int i = 0; i < num_streams; i++) {
        arrays[i] = static_cast<float*>(pool.allocate(array_size * sizeof(float), streams[i]));
        
        // Launch kernel on the same stream
        dim3 block(256);
        dim3 grid((array_size + block.x - 1) / block.x);
        
        initialize_kernel<<<grid, block, 0, streams[i]>>>(arrays[i], array_size, i);
    }
    
    // Synchronize and deallocate
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        pool.deallocate(arrays[i], streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    printf("Pool used memory: %zu bytes\n", pool.get_used_memory());
    printf("Pool reserved memory: %zu bytes\n", pool.get_reserved_memory());
}

__global__ void initialize_kernel(float* array, size_t n, int stream_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = stream_id * 1000.0f + idx;
    }
}
```

### Fragmentation Handling

```cpp
// Memory fragmentation analyzer
class MemoryFragmentationAnalyzer {
private:
    struct AllocationInfo {
        void* ptr;
        size_t size;
        std::chrono::high_resolution_clock::time_point alloc_time;
        bool is_active;
    };
    
    std::vector<AllocationInfo> allocations;
    size_t total_allocated;
    size_t peak_allocated;
    
public:
    MemoryFragmentationAnalyzer() : total_allocated(0), peak_allocated(0) {}
    
    void record_allocation(void* ptr, size_t size) {
        AllocationInfo info;
        info.ptr = ptr;
        info.size = size;
        info.alloc_time = std::chrono::high_resolution_clock::now();
        info.is_active = true;
        
        allocations.push_back(info);
        total_allocated += size;
        peak_allocated = std::max(peak_allocated, total_allocated);
    }
    
    void record_deallocation(void* ptr) {
        auto it = std::find_if(allocations.begin(), allocations.end(),
                             [ptr](const AllocationInfo& info) {
                                 return info.ptr == ptr && info.is_active;
                             });
        
        if (it != allocations.end()) {
            it->is_active = false;
            total_allocated -= it->size;
        }
    }
    
    float calculate_fragmentation() {
        if (allocations.empty()) return 0.0f;
        
        // Sort active allocations by address
        std::vector<AllocationInfo*> active_allocs;
        for (auto& alloc : allocations) {
            if (alloc.is_active) {
                active_allocs.push_back(&alloc);
            }
        }
        
        std::sort(active_allocs.begin(), active_allocs.end(),
                 [](const AllocationInfo* a, const AllocationInfo* b) {
                     return a->ptr < b->ptr;
                 });
        
        // Calculate gaps between allocations
        size_t total_gaps = 0;
        for (size_t i = 1; i < active_allocs.size(); i++) {
            char* prev_end = static_cast<char*>(active_allocs[i-1]->ptr) + active_allocs[i-1]->size;
            char* curr_start = static_cast<char*>(active_allocs[i]->ptr);
            
            if (curr_start > prev_end) {
                total_gaps += curr_start - prev_end;
            }
        }
        
        return total_allocated > 0 ? static_cast<float>(total_gaps) / total_allocated : 0.0f;
    }
    
    void print_analysis() {
        printf("Memory Analysis:\n");
        printf("  Total allocated: %zu bytes\n", total_allocated);
        printf("  Peak allocated: %zu bytes\n", peak_allocated);
        printf("  Active allocations: %zu\n", 
               std::count_if(allocations.begin(), allocations.end(),
                           [](const AllocationInfo& info) { return info.is_active; }));
        printf("  Fragmentation ratio: %.2f%%\n", calculate_fragmentation() * 100);
    }
};

// Anti-fragmentation allocator
class DefragmentingAllocator {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool is_free;
        std::chrono::high_resolution_clock::time_point last_used;
    };
    
    std::vector<MemoryBlock> blocks;
    size_t pool_size;
    void* pool_start;
    
public:
    DefragmentingAllocator(size_t size) : pool_size(size) {
        cudaMalloc(&pool_start, pool_size);
        
        // Initialize with one large free block
        MemoryBlock initial_block;
        initial_block.ptr = pool_start;
        initial_block.size = pool_size;
        initial_block.is_free = true;
        initial_block.last_used = std::chrono::high_resolution_clock::now();
        
        blocks.push_back(initial_block);
    }
    
    ~DefragmentingAllocator() {
        cudaFree(pool_start);
    }
    
    void* allocate(size_t size) {
        // First fit algorithm
        for (auto& block : blocks) {
            if (block.is_free && block.size >= size) {
                if (block.size > size) {
                    // Split block
                    MemoryBlock new_block;
                    new_block.ptr = static_cast<char*>(block.ptr) + size;
                    new_block.size = block.size - size;
                    new_block.is_free = true;
                    new_block.last_used = std::chrono::high_resolution_clock::now();
                    
                    blocks.push_back(new_block);
                }
                
                block.size = size;
                block.is_free = false;
                block.last_used = std::chrono::high_resolution_clock::now();
                
                return block.ptr;
            }
        }
        
        // No suitable block found, try defragmentation
        defragment();
        
        // Try again after defragmentation
        for (auto& block : blocks) {
            if (block.is_free && block.size >= size) {
                // Similar allocation logic
                return block.ptr;
            }
        }
        
        return nullptr; // Out of memory
    }
    
    void deallocate(void* ptr) {
        auto it = std::find_if(blocks.begin(), blocks.end(),
                             [ptr](const MemoryBlock& block) {
                                 return block.ptr == ptr && !block.is_free;
                             });
        
        if (it != blocks.end()) {
            it->is_free = true;
            
            // Coalesce adjacent free blocks
            coalesce_free_blocks();
        }
    }
    
private:
    void defragment() {
        // Move all used blocks to the beginning
        std::vector<MemoryBlock> used_blocks;
        for (const auto& block : blocks) {
            if (!block.is_free) {
                used_blocks.push_back(block);
            }
        }
        
        // Sort by last used time (move least recently used first)
        std::sort(used_blocks.begin(), used_blocks.end(),
                 [](const MemoryBlock& a, const MemoryBlock& b) {
                     return a.last_used < b.last_used;
                 });
        
        // Compact memory by moving blocks
        char* current_pos = static_cast<char*>(pool_start);
        
        for (auto& block : used_blocks) {
            if (block.ptr != current_pos) {
                // Move block data
                cudaMemcpy(current_pos, block.ptr, block.size, cudaMemcpyDeviceToDevice);
                block.ptr = current_pos;
            }
            current_pos += block.size;
        }
        
        // Update blocks list
        blocks = used_blocks;
        
        // Add remaining space as one large free block
        if (current_pos < static_cast<char*>(pool_start) + pool_size) {
            MemoryBlock free_block;
            free_block.ptr = current_pos;
            free_block.size = pool_size - (current_pos - static_cast<char*>(pool_start));
            free_block.is_free = true;
            free_block.last_used = std::chrono::high_resolution_clock::now();
            
            blocks.push_back(free_block);
        }
    }
    
    void coalesce_free_blocks() {
        // Sort blocks by address
        std::sort(blocks.begin(), blocks.end(),
                 [](const MemoryBlock& a, const MemoryBlock& b) {
                     return a.ptr < b.ptr;
                 });
        
        // Merge adjacent free blocks
        for (size_t i = 0; i < blocks.size() - 1; ) {
            if (blocks[i].is_free && blocks[i + 1].is_free) {
                char* end_of_first = static_cast<char*>(blocks[i].ptr) + blocks[i].size;
                if (end_of_first == blocks[i + 1].ptr) {
                    blocks[i].size += blocks[i + 1].size;
                    blocks.erase(blocks.begin() + i + 1);
                    continue;
                }
            }
            i++;
        }
    }
};
```

## Performance Optimization Examples

```cpp
// Memory bandwidth benchmark
class MemoryBandwidthBenchmark {
public:
    void run_all_benchmarks() {
        benchmark_global_memory();
        benchmark_shared_memory();
        benchmark_texture_memory();
        benchmark_constant_memory();
        benchmark_unified_memory();
    }
    
private:
    void benchmark_global_memory() {
        const size_t size = 1024 * 1024 * 1024; // 1GB
        float *d_data, *d_result;
        
        cudaMalloc(&d_data, size);
        cudaMalloc(&d_result, size);
        
        // Initialize data
        cudaMemset(d_data, 1, size);
        
        // Benchmark coalesced access
        auto start = std::chrono::high_resolution_clock::now();
        
        dim3 block(256);
        dim3 grid((size / sizeof(float) + block.x - 1) / block.x);
        
        for (int i = 0; i < 100; i++) {
            coalesced_copy_kernel<<<grid, block>>>(d_data, d_result, size / sizeof(float));
        }
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        float bandwidth = (size * 2 * 100) / (duration.count() * 1e6); // GB/s
        printf("Global memory bandwidth (coalesced): %.2f GB/s\n", bandwidth);
        
        cudaFree(d_data);
        cudaFree(d_result);
    }
    
    void benchmark_unified_memory() {
        const size_t size = 1024 * 1024 * 1024;
        float *data;
        
        cudaMallocManaged(&data, size);
        
        // Benchmark with prefetching
        auto start = std::chrono::high_resolution_clock::now();
        
        cudaMemPrefetchAsync(data, size, 0);
        
        dim3 block(256);
        dim3 grid((size / sizeof(float) + block.x - 1) / block.x);
        
        for (int i = 0; i < 100; i++) {
            unified_memory_kernel<<<grid, block>>>(data, size / sizeof(float));
        }
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        float bandwidth = (size * 100) / (duration.count() * 1e6);
        printf("Unified memory bandwidth (with prefetch): %.2f GB/s\n", bandwidth);
        
        cudaFree(data);
    }
};

__global__ void coalesced_copy_kernel(float* input, float* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

__global__ void unified_memory_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

## Exercises

1. **Unified Memory Optimization**: Implement an adaptive unified memory manager that optimizes data placement based on access patterns.

2. **Texture Memory Applications**: Create a texture-based image processing pipeline with different filtering modes.

3. **Memory Pool Design**: Design and implement a custom memory pool allocator with anti-fragmentation capabilities.

4. **Constant Memory Usage**: Implement a matrix library that efficiently uses constant memory for small matrices.

5. **Memory Bandwidth Analysis**: Analyze and optimize memory access patterns for different algorithms.

## Key Takeaways

- Unified memory provides convenience but requires careful optimization for best performance
- Texture memory offers automatic interpolation and caching benefits for spatial data
- Constant memory is ideal for read-only data accessed by all threads
- Memory pools can significantly reduce allocation overhead and fragmentation
- Understanding memory access patterns is crucial for optimization

## Next Steps

Proceed to [Heterogeneous Programming](10_Heterogeneous_Programming.md) to learn about coordinating CPU and GPU computation in complex applications.
