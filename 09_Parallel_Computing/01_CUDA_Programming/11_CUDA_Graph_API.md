# CUDA Graph API

*Duration: 1 week*

## Overview

CUDA Graphs provide a mechanism to define and execute a series of CUDA operations as a single unit, reducing kernel launch overhead and enabling better optimization opportunities. This section covers graph creation, execution, optimization, and advanced patterns.

## Graph Creation and Execution

### Basic Graph Operations

```cpp
#include <cuda_runtime.h>
#include <vector>
#include <memory>

// CUDA Graph wrapper class
class CudaGraph {
private:
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    std::vector<cudaGraphNode_t> nodes;
    bool is_instantiated;
    
public:
    CudaGraph() : graph(nullptr), graph_exec(nullptr), is_instantiated(false) {
        cudaGraphCreate(&graph, 0);
    }
    
    ~CudaGraph() {
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
    
    // Add kernel node to graph
    cudaGraphNode_t add_kernel_node(const cudaKernelNodeParams& kernel_params,
                                   const std::vector<cudaGraphNode_t>& dependencies = {}) {
        cudaGraphNode_t node;
        cudaGraphAddKernelNode(&node, graph, dependencies.data(), dependencies.size(), &kernel_params);
        nodes.push_back(node);
        return node;
    }
    
    // Add memory copy node to graph
    cudaGraphNode_t add_memcpy_node(void* dst, const void* src, size_t size, cudaMemcpyKind kind,
                                   const std::vector<cudaGraphNode_t>& dependencies = {}) {
        cudaGraphNode_t node;
        cudaMemcpy3DParms memcpy_params = {};
        memcpy_params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), size, 1, 1);
        memcpy_params.dstPtr = make_cudaPitchedPtr(dst, size, 1, 1);
        memcpy_params.extent = make_cudaExtent(size, 1, 1);
        memcpy_params.kind = kind;
        
        cudaGraphAddMemcpyNode(&node, graph, dependencies.data(), dependencies.size(), &memcpy_params);
        nodes.push_back(node);
        return node;
    }
    
    // Add memory set node to graph
    cudaGraphNode_t add_memset_node(void* ptr, int value, size_t size,
                                   const std::vector<cudaGraphNode_t>& dependencies = {}) {
        cudaGraphNode_t node;
        cudaMemsetParams memset_params = {};
        memset_params.dst = ptr;
        memset_params.value = value;
        memset_params.pitch = 0;
        memset_params.elementSize = 1;
        memset_params.width = size;
        memset_params.height = 1;
        
        cudaGraphAddMemsetNode(&node, graph, dependencies.data(), dependencies.size(), &memset_params);
        nodes.push_back(node);
        return node;
    }
    
    // Add host function node to graph
    cudaGraphNode_t add_host_node(cudaHostFn_t host_func, void* user_data,
                                 const std::vector<cudaGraphNode_t>& dependencies = {}) {
        cudaGraphNode_t node;
        cudaHostNodeParams host_params = {};
        host_params.fn = host_func;
        host_params.userData = user_data;
        
        cudaGraphAddHostNode(&node, graph, dependencies.data(), dependencies.size(), &host_params);
        nodes.push_back(node);
        return node;
    }
    
    // Instantiate graph for execution
    void instantiate() {
        if (!is_instantiated) {
            cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
            is_instantiated = true;
        }
    }
    
    // Launch the graph
    void launch(cudaStream_t stream = 0) {
        if (!is_instantiated) {
            instantiate();
        }
        cudaGraphLaunch(graph_exec, stream);
    }
    
    // Update kernel node parameters
    void update_kernel_node(cudaGraphNode_t node, const cudaKernelNodeParams& new_params) {
        if (is_instantiated) {
            cudaGraphExecKernelNodeSetParams(graph_exec, node, &new_params);
        } else {
            cudaGraphKernelNodeSetParams(node, &new_params);
        }
    }
    
    // Update memory copy node
    void update_memcpy_node(cudaGraphNode_t node, void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
        cudaMemcpy3DParms memcpy_params = {};
        memcpy_params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), size, 1, 1);
        memcpy_params.dstPtr = make_cudaPitchedPtr(dst, size, 1, 1);
        memcpy_params.extent = make_cudaExtent(size, 1, 1);
        memcpy_params.kind = kind;
        
        if (is_instantiated) {
            cudaGraphExecMemcpyNodeSetParams(graph_exec, node, &memcpy_params);
        } else {
            cudaGraphMemcpyNodeSetParams(node, &memcpy_params);
        }
    }
    
    // Clone graph for different parameters
    std::unique_ptr<CudaGraph> clone() {
        auto cloned = std::make_unique<CudaGraph>();
        cudaGraphClone(&cloned->graph, graph);
        return cloned;
    }
    
    // Get graph as DOT format for visualization
    void export_to_dot(const std::string& filename) {
        FILE* file = fopen(filename.c_str(), "w");
        if (file) {
            cudaGraphDebugDotPrint(graph, filename.c_str(), 0);
            fclose(file);
        }
    }
};

// Stream capture for automatic graph creation
class StreamCapture {
private:
    cudaStream_t stream;
    cudaGraph_t captured_graph;
    cudaGraphExec_t graph_exec;
    bool is_capturing;
    bool is_instantiated;
    
public:
    StreamCapture() : stream(nullptr), captured_graph(nullptr), graph_exec(nullptr), 
                     is_capturing(false), is_instantiated(false) {
        cudaStreamCreate(&stream);
    }
    
    ~StreamCapture() {
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (captured_graph) {
            cudaGraphDestroy(captured_graph);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    
    void begin_capture() {
        if (!is_capturing) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            is_capturing = true;
        }
    }
    
    void end_capture() {
        if (is_capturing) {
            cudaStreamEndCapture(stream, &captured_graph);
            is_capturing = false;
        }
    }
    
    void instantiate() {
        if (!is_instantiated && captured_graph) {
            cudaGraphInstantiate(&graph_exec, captured_graph, nullptr, nullptr, 0);
            is_instantiated = true;
        }
    }
    
    void launch(cudaStream_t launch_stream = 0) {
        if (!is_instantiated) {
            instantiate();
        }
        if (graph_exec) {
            cudaGraphLaunch(graph_exec, launch_stream);
        }
    }
    
    cudaStream_t get_stream() const { return stream; }
    
    // Execute captured operations
    template<typename F>
    void capture_and_execute(F&& operations, int num_executions = 1) {
        begin_capture();
        operations(stream);
        end_capture();
        instantiate();
        
        for (int i = 0; i < num_executions; i++) {
            launch();
            cudaStreamSynchronize(stream);
        }
    }
};

// Example kernels for graph operations
__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_scale_kernel(float* a, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scale;
    }
}

__global__ void vector_sqrt_kernel(float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = sqrtf(a[idx]);
    }
}

// Basic graph example
void basic_graph_example() {
    const int n = 1000000;
    const size_t bytes = n * sizeof(float);
    
    // Allocate host and device memory
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    h_a = new float[n];
    h_b = new float[n];
    h_c = new float[n];
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Create graph
    CudaGraph graph;
    
    // Add memory copy nodes
    auto copy_a_node = graph.add_memcpy_node(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    auto copy_b_node = graph.add_memcpy_node(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Add kernel node
    cudaKernelNodeParams kernel_params = {};
    void* kernel_args[] = {&d_a, &d_b, &d_c, &n};
    kernel_params.func = (void*)vector_add_kernel;
    kernel_params.gridDim = dim3((n + 255) / 256);
    kernel_params.blockDim = dim3(256);
    kernel_params.sharedMemBytes = 0;
    kernel_params.kernelParams = kernel_args;
    kernel_params.extra = nullptr;
    
    auto kernel_node = graph.add_kernel_node(kernel_params, {copy_a_node, copy_b_node});
    
    // Add copy result back to host
    auto copy_result_node = graph.add_memcpy_node(h_c, d_c, bytes, cudaMemcpyDeviceToHost, {kernel_node});
    
    // Execute graph
    graph.launch();
    cudaDeviceSynchronize();
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < n && correct; i++) {
        if (abs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
        }
    }
    
    printf("Graph execution %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

### Stream Capture Examples

```cpp
// Stream capture for complex operations
void stream_capture_example() {
    const int n = 1000000;
    const size_t bytes = n * sizeof(float);
    
    float *h_data, *d_data;
    h_data = new float[n];
    cudaMalloc(&d_data, bytes);
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_data[i] = static_cast<float>(i + 1);
    }
    
    StreamCapture capture;
    
    // Capture a sequence of operations
    capture.capture_and_execute([&](cudaStream_t stream) {
        // Copy data to device
        cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream);
        
        // Launch multiple kernels
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        // Scale by 2
        vector_scale_kernel<<<grid, block, 0, stream>>>(d_data, 2.0f, n);
        
        // Take square root
        vector_sqrt_kernel<<<grid, block, 0, stream>>>(d_data, n);
        
        // Scale by 0.5
        vector_scale_kernel<<<grid, block, 0, stream>>>(d_data, 0.5f, n);
        
        // Copy result back
        cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, stream);
        
    }, 10); // Execute 10 times
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < n && correct; i++) {
        float expected = sqrtf(static_cast<float>(i + 1) * 2.0f) * 0.5f;
        if (abs(h_data[i] - expected) > 1e-5) {
            correct = false;
        }
    }
    
    printf("Stream capture execution %s\n", correct ? "PASSED" : "FAILED");
    
    delete[] h_data;
    cudaFree(d_data);
}
```

## Graph Optimization and Performance

### Conditional Execution

```cpp
// Conditional graph execution
class ConditionalGraph {
private:
    cudaGraph_t main_graph;
    cudaGraphExec_t main_exec;
    cudaGraph_t true_subgraph;
    cudaGraphExec_t true_exec;
    cudaGraph_t false_subgraph;
    cudaGraphExec_t false_exec;
    
public:
    ConditionalGraph() {
        cudaGraphCreate(&main_graph, 0);
        cudaGraphCreate(&true_subgraph, 0);
        cudaGraphCreate(&false_subgraph, 0);
    }
    
    ~ConditionalGraph() {
        if (main_exec) cudaGraphExecDestroy(main_exec);
        if (true_exec) cudaGraphExecDestroy(true_exec);
        if (false_exec) cudaGraphExecDestroy(false_exec);
        
        cudaGraphDestroy(main_graph);
        cudaGraphDestroy(true_subgraph);
        cudaGraphDestroy(false_subgraph);
    }
    
    void build_conditional_graph(float* data, int n, float threshold) {
        // Create conditional node
        cudaGraphNode_t conditional_node;
        cudaConditionalNodeParams conditional_params = {};
        
        // This would be a custom conditional implementation
        // For demonstration, we'll use a simpler approach
        
        // Build true branch (data > threshold)
        build_true_branch(data, n, threshold);
        
        // Build false branch (data <= threshold)
        build_false_branch(data, n, threshold);
        
        // Instantiate graphs
        cudaGraphInstantiate(&true_exec, true_subgraph, nullptr, nullptr, 0);
        cudaGraphInstantiate(&false_exec, false_subgraph, nullptr, nullptr, 0);
    }
    
    void execute_conditional(cudaStream_t stream, bool condition) {
        if (condition) {
            cudaGraphLaunch(true_exec, stream);
        } else {
            cudaGraphLaunch(false_exec, stream);
        }
    }
    
private:
    void build_true_branch(float* data, int n, float threshold) {
        // Add kernel for true condition
        cudaKernelNodeParams kernel_params = {};
        void* args[] = {&data, &n};
        kernel_params.func = (void*)true_branch_kernel;
        kernel_params.gridDim = dim3((n + 255) / 256);
        kernel_params.blockDim = dim3(256);
        kernel_params.kernelParams = args;
        
        cudaGraphNode_t node;
        cudaGraphAddKernelNode(&node, true_subgraph, nullptr, 0, &kernel_params);
    }
    
    void build_false_branch(float* data, int n, float threshold) {
        // Add kernel for false condition
        cudaKernelNodeParams kernel_params = {};
        void* args[] = {&data, &n};
        kernel_params.func = (void*)false_branch_kernel;
        kernel_params.gridDim = dim3((n + 255) / 256);
        kernel_params.blockDim = dim3(256);
        kernel_params.kernelParams = args;
        
        cudaGraphNode_t node;
        cudaGraphAddKernelNode(&node, false_subgraph, nullptr, 0, &kernel_params);
    }
};

__global__ void true_branch_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void false_branch_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 0.5f;
    }
}
```

### Graph Templates and Parameterization

```cpp
// Template-based graph system
template<typename T>
class ParameterizedGraph {
private:
    struct GraphTemplate {
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        std::vector<cudaGraphNode_t> parameterizable_nodes;
        std::map<std::string, cudaGraphNode_t> named_nodes;
        bool is_instantiated;
        
        GraphTemplate() : graph(nullptr), graph_exec(nullptr), is_instantiated(false) {
            cudaGraphCreate(&graph, 0);
        }
        
        ~GraphTemplate() {
            if (graph_exec) cudaGraphExecDestroy(graph_exec);
            if (graph) cudaGraphDestroy(graph);
        }
    };
    
    std::unique_ptr<GraphTemplate> graph_template;
    
public:
    ParameterizedGraph() : graph_template(std::make_unique<GraphTemplate>()) {}
    
    void create_template(int max_size) {
        // Create a template graph with placeholder parameters
        T* dummy_data = nullptr;
        int dummy_size = max_size;
        
        // Add parameterizable kernel nodes
        for (int stage = 0; stage < 3; stage++) {
            cudaKernelNodeParams kernel_params = {};
            void* args[] = {&dummy_data, &dummy_size};
            
            switch (stage) {
                case 0:
                    kernel_params.func = (void*)process_stage1_kernel<T>;
                    break;
                case 1:
                    kernel_params.func = (void*)process_stage2_kernel<T>;
                    break;
                case 2:
                    kernel_params.func = (void*)process_stage3_kernel<T>;
                    break;
            }
            
            kernel_params.gridDim = dim3((max_size + 255) / 256);
            kernel_params.blockDim = dim3(256);
            kernel_params.kernelParams = args;
            
            cudaGraphNode_t node;
            std::vector<cudaGraphNode_t> deps;
            if (stage > 0) {
                deps.push_back(graph_template->parameterizable_nodes.back());
            }
            
            cudaGraphAddKernelNode(&node, graph_template->graph, deps.data(), deps.size(), &kernel_params);
            graph_template->parameterizable_nodes.push_back(node);
            graph_template->named_nodes[std::string("stage") + std::to_string(stage)] = node;
        }
    }
    
    void instantiate_with_parameters(T* data, int size) {
        // Update all parameterizable nodes with actual parameters
        for (int i = 0; i < graph_template->parameterizable_nodes.size(); i++) {
            cudaKernelNodeParams kernel_params = {};
            void* args[] = {&data, &size};
            
            switch (i) {
                case 0:
                    kernel_params.func = (void*)process_stage1_kernel<T>;
                    break;
                case 1:
                    kernel_params.func = (void*)process_stage2_kernel<T>;
                    break;
                case 2:
                    kernel_params.func = (void*)process_stage3_kernel<T>;
                    break;
            }
            
            kernel_params.gridDim = dim3((size + 255) / 256);
            kernel_params.blockDim = dim3(256);
            kernel_params.kernelParams = args;
            
            cudaGraphKernelNodeSetParams(graph_template->parameterizable_nodes[i], &kernel_params);
        }
        
        // Instantiate the graph
        if (!graph_template->is_instantiated) {
            cudaGraphInstantiate(&graph_template->graph_exec, graph_template->graph, nullptr, nullptr, 0);
            graph_template->is_instantiated = true;
        }
    }
    
    void execute(cudaStream_t stream = 0) {
        if (graph_template->is_instantiated) {
            cudaGraphLaunch(graph_template->graph_exec, stream);
        }
    }
    
    void update_node_parameters(const std::string& node_name, T* new_data, int new_size) {
        auto it = graph_template->named_nodes.find(node_name);
        if (it != graph_template->named_nodes.end()) {
            cudaKernelNodeParams kernel_params = {};
            void* args[] = {&new_data, &new_size};
            
            // Determine kernel function based on node name
            if (node_name == "stage0") {
                kernel_params.func = (void*)process_stage1_kernel<T>;
            } else if (node_name == "stage1") {
                kernel_params.func = (void*)process_stage2_kernel<T>;
            } else if (node_name == "stage2") {
                kernel_params.func = (void*)process_stage3_kernel<T>;
            }
            
            kernel_params.gridDim = dim3((new_size + 255) / 256);
            kernel_params.blockDim = dim3(256);
            kernel_params.kernelParams = args;
            
            if (graph_template->is_instantiated) {
                cudaGraphExecKernelNodeSetParams(graph_template->graph_exec, it->second, &kernel_params);
            } else {
                cudaGraphKernelNodeSetParams(it->second, &kernel_params);
            }
        }
    }
};

// Template kernel functions
template<typename T>
__global__ void process_stage1_kernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + static_cast<T>(1);
    }
}

template<typename T>
__global__ void process_stage2_kernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * static_cast<T>(2);
    }
}

template<typename T>
__global__ void process_stage3_kernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrt(data[idx]);
    }
}

// Usage example
void parameterized_graph_example() {
    const int n = 100000;
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    // Initialize data
    std::vector<float> h_data(n);
    for (int i = 0; i < n; i++) {
        h_data[i] = static_cast<float>(i + 1);
    }
    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create parameterized graph
    ParameterizedGraph<float> param_graph;
    param_graph.create_template(n);
    param_graph.instantiate_with_parameters(d_data, n);
    
    // Execute graph
    param_graph.execute();
    cudaDeviceSynchronize();
    
    // Update parameters and execute again
    param_graph.update_node_parameters("stage1", d_data, n/2);
    param_graph.execute();
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
}
```

## Advanced Graph Patterns

### Nested Graphs and Subgraphs

```cpp
// Hierarchical graph system
class HierarchicalGraph {
private:
    struct SubGraph {
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        std::string name;
        std::vector<cudaGraphNode_t> input_nodes;
        std::vector<cudaGraphNode_t> output_nodes;
        bool is_instantiated;
        
        SubGraph(const std::string& n) : name(n), graph(nullptr), graph_exec(nullptr), is_instantiated(false) {
            cudaGraphCreate(&graph, 0);
        }
        
        ~SubGraph() {
            if (graph_exec) cudaGraphExecDestroy(graph_exec);
            if (graph) cudaGraphDestroy(graph);
        }
    };
    
    cudaGraph_t main_graph;
    cudaGraphExec_t main_exec;
    std::vector<std::unique_ptr<SubGraph>> subgraphs;
    std::map<std::string, size_t> subgraph_indices;
    bool is_instantiated;
    
public:
    HierarchicalGraph() : main_graph(nullptr), main_exec(nullptr), is_instantiated(false) {
        cudaGraphCreate(&main_graph, 0);
    }
    
    ~HierarchicalGraph() {
        if (main_exec) cudaGraphExecDestroy(main_exec);
        if (main_graph) cudaGraphDestroy(main_graph);
    }
    
    size_t create_subgraph(const std::string& name) {
        auto subgraph = std::make_unique<SubGraph>(name);
        size_t index = subgraphs.size();
        subgraphs.push_back(std::move(subgraph));
        subgraph_indices[name] = index;
        return index;
    }
    
    void add_kernel_to_subgraph(size_t subgraph_index, const cudaKernelNodeParams& kernel_params,
                               const std::vector<cudaGraphNode_t>& dependencies = {}) {
        if (subgraph_index >= subgraphs.size()) return;
        
        cudaGraphNode_t node;
        cudaGraphAddKernelNode(&node, subgraphs[subgraph_index]->graph, 
                              dependencies.data(), dependencies.size(), &kernel_params);
        
        // Track as output node if no other nodes depend on it
        subgraphs[subgraph_index]->output_nodes.push_back(node);
    }
    
    void add_memcpy_to_subgraph(size_t subgraph_index, void* dst, const void* src, size_t size, 
                               cudaMemcpyKind kind, const std::vector<cudaGraphNode_t>& dependencies = {}) {
        if (subgraph_index >= subgraphs.size()) return;
        
        cudaGraphNode_t node;
        cudaMemcpy3DParms memcpy_params = {};
        memcpy_params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), size, 1, 1);
        memcpy_params.dstPtr = make_cudaPitchedPtr(dst, size, 1, 1);
        memcpy_params.extent = make_cudaExtent(size, 1, 1);
        memcpy_params.kind = kind;
        
        cudaGraphAddMemcpyNode(&node, subgraphs[subgraph_index]->graph,
                              dependencies.data(), dependencies.size(), &memcpy_params);
        
        if (dependencies.empty()) {
            subgraphs[subgraph_index]->input_nodes.push_back(node);
        } else {
            subgraphs[subgraph_index]->output_nodes.push_back(node);
        }
    }
    
    void compose_subgraphs() {
        // Instantiate all subgraphs
        for (auto& subgraph : subgraphs) {
            if (!subgraph->is_instantiated) {
                cudaGraphInstantiate(&subgraph->graph_exec, subgraph->graph, nullptr, nullptr, 0);
                subgraph->is_instantiated = true;
            }
        }
        
        // Add child graph nodes to main graph
        std::vector<cudaGraphNode_t> child_nodes;
        for (auto& subgraph : subgraphs) {
            cudaGraphNode_t child_node;
            cudaGraphAddChildGraphNode(&child_node, main_graph, nullptr, 0, subgraph->graph);
            child_nodes.push_back(child_node);
        }
        
        // Add dependencies between child graphs if needed
        for (size_t i = 1; i < child_nodes.size(); i++) {
            cudaGraphAddDependencies(main_graph, &child_nodes[i-1], &child_nodes[i], 1);
        }
    }
    
    void instantiate_main_graph() {
        if (!is_instantiated) {
            cudaGraphInstantiate(&main_exec, main_graph, nullptr, nullptr, 0);
            is_instantiated = true;
        }
    }
    
    void execute(cudaStream_t stream = 0) {
        if (!is_instantiated) {
            instantiate_main_graph();
        }
        cudaGraphLaunch(main_exec, stream);
    }
    
    void execute_subgraph(const std::string& name, cudaStream_t stream = 0) {
        auto it = subgraph_indices.find(name);
        if (it != subgraph_indices.end()) {
            auto& subgraph = subgraphs[it->second];
            if (!subgraph->is_instantiated) {
                cudaGraphInstantiate(&subgraph->graph_exec, subgraph->graph, nullptr, nullptr, 0);
                subgraph->is_instantiated = true;
            }
            cudaGraphLaunch(subgraph->graph_exec, stream);
        }
    }
};

// Dynamic graph modification
class DynamicGraph {
private:
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    std::vector<cudaGraphNode_t> nodes;
    std::map<std::string, cudaGraphNode_t> named_nodes;
    bool is_instantiated;
    
public:
    DynamicGraph() : graph(nullptr), graph_exec(nullptr), is_instantiated(false) {
        cudaGraphCreate(&graph, 0);
    }
    
    ~DynamicGraph() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
    }
    
    void add_named_kernel(const std::string& name, const cudaKernelNodeParams& kernel_params,
                         const std::vector<std::string>& dependency_names = {}) {
        std::vector<cudaGraphNode_t> dependencies;
        for (const auto& dep_name : dependency_names) {
            auto it = named_nodes.find(dep_name);
            if (it != named_nodes.end()) {
                dependencies.push_back(it->second);
            }
        }
        
        cudaGraphNode_t node;
        cudaGraphAddKernelNode(&node, graph, dependencies.data(), dependencies.size(), &kernel_params);
        
        nodes.push_back(node);
        named_nodes[name] = node;
    }
    
    void remove_node(const std::string& name) {
        auto it = named_nodes.find(name);
        if (it != named_nodes.end()) {
            cudaGraphRemoveNode(graph, it->second);
            named_nodes.erase(it);
            
            // Remove from nodes vector
            nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                                     [node = it->second](cudaGraphNode_t n) { return n == node; }),
                       nodes.end());
            
            // Need to re-instantiate
            if (is_instantiated) {
                cudaGraphExecDestroy(graph_exec);
                graph_exec = nullptr;
                is_instantiated = false;
            }
        }
    }
    
    void add_dependency(const std::string& from_node, const std::string& to_node) {
        auto from_it = named_nodes.find(from_node);
        auto to_it = named_nodes.find(to_node);
        
        if (from_it != named_nodes.end() && to_it != named_nodes.end()) {
            cudaGraphAddDependencies(graph, &from_it->second, &to_it->second, 1);
            
            // Need to re-instantiate
            if (is_instantiated) {
                cudaGraphExecDestroy(graph_exec);
                graph_exec = nullptr;
                is_instantiated = false;
            }
        }
    }
    
    void remove_dependency(const std::string& from_node, const std::string& to_node) {
        auto from_it = named_nodes.find(from_node);
        auto to_it = named_nodes.find(to_node);
        
        if (from_it != named_nodes.end() && to_it != named_nodes.end()) {
            cudaGraphRemoveDependencies(graph, &from_it->second, &to_it->second, 1);
            
            // Need to re-instantiate
            if (is_instantiated) {
                cudaGraphExecDestroy(graph_exec);
                graph_exec = nullptr;
                is_instantiated = false;
            }
        }
    }
    
    void instantiate() {
        if (!is_instantiated) {
            cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
            is_instantiated = true;
        }
    }
    
    void execute(cudaStream_t stream = 0) {
        if (!is_instantiated) {
            instantiate();
        }
        cudaGraphLaunch(graph_exec, stream);
    }
    
    void update_kernel_parameters(const std::string& name, const cudaKernelNodeParams& new_params) {
        auto it = named_nodes.find(name);
        if (it != named_nodes.end()) {
            if (is_instantiated) {
                cudaGraphExecKernelNodeSetParams(graph_exec, it->second, &new_params);
            } else {
                cudaGraphKernelNodeSetParams(it->second, &new_params);
            }
        }
    }
    
    void clone_subgraph(const std::vector<std::string>& node_names, DynamicGraph& target_graph) {
        std::set<cudaGraphNode_t> nodes_to_clone;
        for (const auto& name : node_names) {
            auto it = named_nodes.find(name);
            if (it != named_nodes.end()) {
                nodes_to_clone.insert(it->second);
            }
        }
        
        if (!nodes_to_clone.empty()) {
            cudaGraph_t cloned_graph;
            cudaGraphClone(&cloned_graph, graph);
            
            // This would require more complex logic to extract specific nodes
            // For now, we'll just copy the entire graph
            target_graph.graph = cloned_graph;
        }
    }
};

// Example usage of hierarchical graphs
void hierarchical_graph_example() {
    const int n = 100000;
    float *d_data, *d_temp;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_temp, n * sizeof(float));
    
    HierarchicalGraph hierarchical;
    
    // Create preprocessing subgraph
    size_t preprocess_subgraph = hierarchical.create_subgraph("preprocess");
    
    cudaKernelNodeParams preprocess_params = {};
    void* preprocess_args[] = {&d_data, &n};
    preprocess_params.func = (void*)vector_scale_kernel;
    preprocess_params.gridDim = dim3((n + 255) / 256);
    preprocess_params.blockDim = dim3(256);
    preprocess_params.kernelParams = preprocess_args;
    
    hierarchical.add_kernel_to_subgraph(preprocess_subgraph, preprocess_params);
    
    // Create processing subgraph
    size_t process_subgraph = hierarchical.create_subgraph("process");
    
    cudaKernelNodeParams process_params = {};
    void* process_args[] = {&d_data, &n};
    process_params.func = (void*)vector_sqrt_kernel;
    process_params.gridDim = dim3((n + 255) / 256);
    process_params.blockDim = dim3(256);
    process_params.kernelParams = process_args;
    
    hierarchical.add_kernel_to_subgraph(process_subgraph, process_params);
    
    // Compose and execute
    hierarchical.compose_subgraphs();
    hierarchical.execute();
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    cudaFree(d_temp);
}
```

## Performance Comparison

```cpp
// Performance comparison: graphs vs. regular kernel launches
class GraphPerformanceComparison {
private:
    struct BenchmarkResult {
        std::string name;
        float time_ms;
        float speedup;
    };
    
    std::vector<BenchmarkResult> results;
    
public:
    void run_comparison(int n, int num_iterations = 1000) {
        const size_t bytes = n * sizeof(float);
        
        float *h_data, *d_data;
        h_data = new float[n];
        cudaMalloc(&d_data, bytes);
        
        // Initialize data
        for (int i = 0; i < n; i++) {
            h_data[i] = static_cast<float>(i + 1);
        }
        
        // Benchmark regular kernel launches
        float regular_time = benchmark_regular_launches(h_data, d_data, n, bytes, num_iterations);
        
        // Benchmark graph execution
        float graph_time = benchmark_graph_execution(h_data, d_data, n, bytes, num_iterations);
        
        // Benchmark stream capture
        float capture_time = benchmark_stream_capture(h_data, d_data, n, bytes, num_iterations);
        
        // Store results
        results.push_back({"Regular Launches", regular_time, 1.0f});
        results.push_back({"Graph Execution", graph_time, regular_time / graph_time});
        results.push_back({"Stream Capture", capture_time, regular_time / capture_time});
        
        // Print results
        print_results();
        
        delete[] h_data;
        cudaFree(d_data);
    }
    
private:
    float benchmark_regular_launches(float* h_data, float* d_data, int n, size_t bytes, int iterations) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        cudaEventRecord(start);
        
        for (int i = 0; i < iterations; i++) {
            cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
            vector_scale_kernel<<<grid, block>>>(d_data, 2.0f, n);
            vector_sqrt_kernel<<<grid, block>>>(d_data, n);
            vector_scale_kernel<<<grid, block>>>(d_data, 0.5f, n);
            cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return time_ms;
    }
    
    float benchmark_graph_execution(float* h_data, float* d_data, int n, size_t bytes, int iterations) {
        // Create and build graph
        CudaGraph graph;
        
        // Add nodes to graph
        auto copy_to_device = graph.add_memcpy_node(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        
        cudaKernelNodeParams scale_params = {};
        void* scale_args[] = {&d_data, &n};
        scale_params.func = (void*)vector_scale_kernel;
        scale_params.gridDim = dim3((n + 255) / 256);
        scale_params.blockDim = dim3(256);
        scale_params.kernelParams = scale_args;
        
        auto scale_node = graph.add_kernel_node(scale_params, {copy_to_device});
        
        cudaKernelNodeParams sqrt_params = {};
        void* sqrt_args[] = {&d_data, &n};
        sqrt_params.func = (void*)vector_sqrt_kernel;
        sqrt_params.gridDim = dim3((n + 255) / 256);
        sqrt_params.blockDim = dim3(256);
        sqrt_params.kernelParams = sqrt_args;
        
        auto sqrt_node = graph.add_kernel_node(sqrt_params, {scale_node});
        
        cudaKernelNodeParams scale2_params = {};
        void* scale2_args[] = {&d_data, &n};
        scale2_params.func = (void*)vector_scale_kernel;
        scale2_params.gridDim = dim3((n + 255) / 256);
        scale2_params.blockDim = dim3(256);
        scale2_params.kernelParams = scale2_args;
        
        auto scale2_node = graph.add_kernel_node(scale2_params, {sqrt_node});
        
        auto copy_to_host = graph.add_memcpy_node(h_data, d_data, bytes, cudaMemcpyDeviceToHost, {scale2_node});
        
        // Benchmark execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        
        for (int i = 0; i < iterations; i++) {
            graph.launch();
            cudaDeviceSynchronize();
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return time_ms;
    }
    
    float benchmark_stream_capture(float* h_data, float* d_data, int n, size_t bytes, int iterations) {
        StreamCapture capture;
        
        // Capture the sequence once
        capture.begin_capture();
        
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, capture.get_stream());
        vector_scale_kernel<<<grid, block, 0, capture.get_stream()>>>(d_data, 2.0f, n);
        vector_sqrt_kernel<<<grid, block, 0, capture.get_stream()>>>(d_data, n);
        vector_scale_kernel<<<grid, block, 0, capture.get_stream()>>>(d_data, 0.5f, n);
        cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, capture.get_stream());
        
        capture.end_capture();
        capture.instantiate();
        
        // Benchmark execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        
        for (int i = 0; i < iterations; i++) {
            capture.launch();
            cudaDeviceSynchronize();
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return time_ms;
    }
    
    void print_results() {
        printf("\n=== Graph Performance Comparison ===\n");
        printf("Method                | Time (ms)    | Speedup\n");
        printf("---------------------|-------------|--------\n");
        
        for (const auto& result : results) {
            printf("%-20s | %10.2f  | %6.2fx\n", 
                   result.name.c_str(), result.time_ms, result.speedup);
        }
        printf("\n");
    }
};

// Main example function
void cuda_graph_examples() {
    printf("=== CUDA Graph API Examples ===\n");
    
    // Basic graph example
    printf("\n1. Basic Graph Example:\n");
    basic_graph_example();
    
    // Stream capture example
    printf("\n2. Stream Capture Example:\n");
    stream_capture_example();
    
    // Parameterized graph example
    printf("\n3. Parameterized Graph Example:\n");
    parameterized_graph_example();
    
    // Hierarchical graph example
    printf("\n4. Hierarchical Graph Example:\n");
    hierarchical_graph_example();
    
    // Performance comparison
    printf("\n5. Performance Comparison:\n");
    GraphPerformanceComparison comparison;
    comparison.run_comparison(100000, 100);
}
```

## Exercises

1. **Complex Graph Pipeline**: Create a graph that implements a complete image processing pipeline with multiple stages and conditional execution.

2. **Dynamic Graph Modification**: Implement a system that can modify graph structure at runtime based on data characteristics.

3. **Graph Optimization**: Analyze and optimize a graph for different hardware configurations and data sizes.

4. **Nested Graph Hierarchies**: Build a hierarchical graph system that can compose complex workflows from reusable subgraphs.

5. **Performance Analysis**: Compare graph execution performance with regular kernel launches across different scenarios.

## Key Takeaways

- CUDA Graphs reduce kernel launch overhead by batching operations
- Stream capture provides an easy way to create graphs from existing code
- Graph templates enable parameterized execution with runtime updates
- Hierarchical graphs allow composition of complex workflows
- Conditional execution and dynamic modification enable adaptive algorithms
- Performance benefits are most significant for workloads with many small kernels

## Next Steps

Proceed to [Advanced Topics and Emerging Trends](12_Advanced_Topics_and_Emerging_Trends.md) to explore cutting-edge CUDA features and future directions in GPU computing.
