# Deep Learning with CUDA

*Duration: 3 weeks*

## Overview

CUDA provides the foundation for modern deep learning acceleration. This section covers cuDNN for neural network primitives, TensorRT for inference optimization, NCCL for multi-GPU communication, and custom CUDA kernel development for machine learning applications.

## cuDNN (CUDA Deep Neural Network Library)

### Architecture and Capabilities

cuDNN provides GPU-accelerated primitives for deep neural networks:

```cpp
#include <cudnn.h>

class CuDNNManager {
private:
    cudnnHandle_t cudnn_handle;
    
public:
    CuDNNManager() {
        cudnnCreate(&cudnn_handle);
        
        // Set math type for tensor cores (if available)
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        if (prop.major >= 7) { // Tensor cores available on Volta and newer
            cudnnSetMathType(cudnn_handle, CUDNN_TENSOR_OP_MATH);
        }
    }
    
    ~CuDNNManager() {
        cudnnDestroy(cudnn_handle);
    }
    
    cudnnHandle_t get_handle() { return cudnn_handle; }
};

// Tensor descriptor wrapper
class TensorDescriptor {
private:
    cudnnTensorDescriptor_t desc;
    
public:
    TensorDescriptor() {
        cudnnCreateTensorDescriptor(&desc);
    }
    
    ~TensorDescriptor() {
        cudnnDestroyTensorDescriptor(desc);
    }
    
    void set_4d(cudnnDataType_t data_type, int n, int c, int h, int w) {
        cudnnSetTensorNdDescriptor(desc, data_type, 4,
                                  new int[4]{n, c, h, w},
                                  new int[4]{c*h*w, h*w, w, 1});
    }
    
    cudnnTensorDescriptor_t* get() { return &desc; }
};
```

### Convolution Operations

```cpp
// Convolution layer implementation
class ConvolutionLayer {
private:
    cudnnHandle_t cudnn_handle;
    
    // Descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t activation_desc;
    
    // Data pointers
    float *d_input, *d_output, *d_filter, *d_bias;
    float *d_workspace;
    
    // Layer parameters
    int batch_size, input_channels, input_height, input_width;
    int output_channels, output_height, output_width;
    int filter_height, filter_width;
    int pad_h, pad_w, stride_h, stride_w;
    
    size_t workspace_size;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    
public:
    ConvolutionLayer(cudnnHandle_t handle, int batch, int in_c, int in_h, int in_w,
                    int out_c, int filt_h, int filt_w, int pad_h, int pad_w,
                    int str_h, int str_w) :
        cudnn_handle(handle), batch_size(batch), input_channels(in_c),
        input_height(in_h), input_width(in_w), output_channels(out_c),
        filter_height(filt_h), filter_width(filt_w),
        pad_h(pad_h), pad_w(pad_w), stride_h(str_h), stride_w(str_w) {
        
        initialize();
        find_best_algorithms();
        allocate_memory();
    }
    
    void initialize() {
        // Create descriptors
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnCreateFilterDescriptor(&filter_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnCreateActivationDescriptor(&activation_desc);
        
        // Calculate output dimensions
        cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                       1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        
        cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                            &batch_size, &output_channels,
                                            &output_height, &output_width);
        
        // Set tensor descriptors
        cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 4,
                                  new int[4]{batch_size, input_channels, input_height, input_width},
                                  new int[4]{input_channels*input_height*input_width, 
                                           input_height*input_width, input_width, 1});
        
        cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 4,
                                  new int[4]{batch_size, output_channels, output_height, output_width},
                                  new int[4]{output_channels*output_height*output_width,
                                           output_height*output_width, output_width, 1});
        
        cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4,
                                  new int[4]{output_channels, input_channels, filter_height, filter_width});
        
        cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 4,
                                  new int[4]{1, output_channels, 1, 1},
                                  new int[4]{output_channels, 1, 1, 1});
        
        // Set activation descriptor (ReLU)
        cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU,
                                    CUDNN_NOT_PROPAGATE_NAN, 0.0);
    }
    
    void find_best_algorithms() {
        // Find best forward algorithm
        int algo_count;
        cudnnConvolutionFwdAlgoPerf_t fwd_perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        
        cudnnFindConvolutionForwardAlgorithm(cudnn_handle, input_desc, filter_desc,
                                           conv_desc, output_desc,
                                           CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &algo_count, fwd_perf);
        fwd_algo = fwd_perf[0].algo;
        
        // Get workspace size
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc, filter_desc,
                                              conv_desc, output_desc, fwd_algo, &workspace_size);
        
        // Find best backward algorithms
        cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
        cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle, filter_desc, output_desc,
                                                 conv_desc, input_desc,
                                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                                                 &algo_count, bwd_data_perf);
        bwd_data_algo = bwd_data_perf[0].algo;
        
        cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
        cudnnFindConvolutionBackwardFilterAlgorithm(cudnn_handle, input_desc, output_desc,
                                                   conv_desc, filter_desc,
                                                   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                                                   &algo_count, bwd_filter_perf);
        bwd_filter_algo = bwd_filter_perf[0].algo;
    }
    
    void forward(float* input, float* output) {
        const float alpha = 1.0f, beta = 0.0f;
        
        // Convolution
        cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, input,
                              filter_desc, d_filter, conv_desc, fwd_algo,
                              d_workspace, workspace_size, &beta, output_desc, output);
        
        // Add bias
        cudnnAddTensor(cudnn_handle, &alpha, bias_desc, d_bias, &alpha, output_desc, output);
        
        // Apply activation
        cudnnActivationForward(cudnn_handle, activation_desc, &alpha, output_desc, output,
                             &beta, output_desc, output);
    }
    
    void backward(float* input, float* grad_output, float* grad_input, float* grad_filter, float* grad_bias) {
        const float alpha = 1.0f, beta = 0.0f;
        
        // Backward activation
        float* temp_grad;
        cudaMalloc(&temp_grad, batch_size * output_channels * output_height * output_width * sizeof(float));
        
        cudnnActivationBackward(cudnn_handle, activation_desc, &alpha, output_desc, d_output,
                              output_desc, grad_output, output_desc, d_output,
                              &beta, output_desc, temp_grad);
        
        // Backward bias
        cudnnConvolutionBackwardBias(cudnn_handle, &alpha, output_desc, temp_grad,
                                   &beta, bias_desc, grad_bias);
        
        // Backward filter
        cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, input_desc, input,
                                     output_desc, temp_grad, conv_desc, bwd_filter_algo,
                                     d_workspace, workspace_size, &beta, filter_desc, grad_filter);
        
        // Backward data
        if (grad_input) {
            cudnnConvolutionBackwardData(cudnn_handle, &alpha, filter_desc, d_filter,
                                       output_desc, temp_grad, conv_desc, bwd_data_algo,
                                       d_workspace, workspace_size, &beta, input_desc, grad_input);
        }
        
        cudaFree(temp_grad);
    }
};
```

### Batch Normalization

```cpp
class BatchNormalizationLayer {
private:
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_desc, output_desc, bn_desc;
    
    float *d_scale, *d_bias, *d_running_mean, *d_running_var;
    float *d_saved_mean, *d_saved_inv_var;
    
    double epsilon;
    double momentum;
    
public:
    BatchNormalizationLayer(cudnnHandle_t handle, int n, int c, int h, int w) :
        cudnn_handle(handle), epsilon(1e-5), momentum(0.1) {
        
        // Create descriptors
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateTensorDescriptor(&bn_desc);
        
        // Set descriptors
        cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 4,
                                  new int[4]{n, c, h, w},
                                  new int[4]{c*h*w, h*w, w, 1});
        
        cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 4,
                                  new int[4]{n, c, h, w},
                                  new int[4]{c*h*w, h*w, w, 1});
        
        // Derive batch norm descriptor
        cudnnDeriveBNTensorDescriptor(bn_desc, input_desc, CUDNN_BATCHNORM_SPATIAL);
        
        // Allocate parameters
        cudaMalloc(&d_scale, c * sizeof(float));
        cudaMalloc(&d_bias, c * sizeof(float));
        cudaMalloc(&d_running_mean, c * sizeof(float));
        cudaMalloc(&d_running_var, c * sizeof(float));
        cudaMalloc(&d_saved_mean, c * sizeof(float));
        cudaMalloc(&d_saved_inv_var, c * sizeof(float));
        
        // Initialize parameters
        initialize_parameters(c);
    }
    
    void forward_training(float* input, float* output) {
        const float alpha = 1.0f, beta = 0.0f;
        
        cudnnBatchNormalizationForwardTraining(
            cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            input_desc, input, output_desc, output,
            bn_desc, d_scale, d_bias, momentum,
            d_running_mean, d_running_var, epsilon,
            d_saved_mean, d_saved_inv_var
        );
    }
    
    void forward_inference(float* input, float* output) {
        const float alpha = 1.0f, beta = 0.0f;
        
        cudnnBatchNormalizationForwardInference(
            cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            input_desc, input, output_desc, output,
            bn_desc, d_scale, d_bias,
            d_running_mean, d_running_var, epsilon
        );
    }
    
    void backward(float* input, float* grad_output, float* grad_input,
                 float* grad_scale, float* grad_bias) {
        const float alpha = 1.0f, beta = 0.0f;
        
        cudnnBatchNormalizationBackward(
            cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, &alpha, &beta,
            input_desc, input, output_desc, grad_output, input_desc, grad_input,
            bn_desc, d_scale, grad_scale, grad_bias, epsilon,
            d_saved_mean, d_saved_inv_var
        );
    }
    
private:
    void initialize_parameters(int channels) {
        // Initialize scale to 1.0
        thrust::device_ptr<float> scale_ptr(d_scale);
        thrust::fill(scale_ptr, scale_ptr + channels, 1.0f);
        
        // Initialize bias to 0.0
        cudaMemset(d_bias, 0, channels * sizeof(float));
        cudaMemset(d_running_mean, 0, channels * sizeof(float));
        
        // Initialize running variance to 1.0
        thrust::device_ptr<float> var_ptr(d_running_var);
        thrust::fill(var_ptr, var_ptr + channels, 1.0f);
    }
};
```

## TensorRT Integration

### Network Definition and Optimization

```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>

class TensorRTEngine {
private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    
    std::vector<void*> bindings;
    std::vector<size_t> binding_sizes;
    cudaStream_t stream;
    
public:
    TensorRTEngine() : runtime(nullptr), engine(nullptr), context(nullptr) {
        cudaStreamCreate(&stream);
    }
    
    ~TensorRTEngine() {
        cleanup();
        cudaStreamDestroy(stream);
    }
    
    bool build_from_onnx(const std::string& onnx_file, int max_batch_size = 1) {
        // Create builder and network
        auto builder = nvinfer1::createInferBuilder(gLogger);
        const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicit_batch);
        
        // Create ONNX parser
        auto parser = nvonnxparser::createParser(*network, gLogger);
        
        // Parse ONNX file
        if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX file: " << onnx_file << std::endl;
            return false;
        }
        
        // Configure builder
        auto config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1ULL << 30); // 1GB
        
        // Enable optimizations
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout << "FP16 optimization enabled" << std::endl;
        }
        
        if (builder->platformHasFastInt8()) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            // Set INT8 calibrator here if needed
            std::cout << "INT8 optimization enabled" << std::endl;
        }
        
        // Build engine
        engine = builder->buildEngineWithConfig(*network, *config);
        if (!engine) {
            std::cerr << "Failed to build TensorRT engine" << std::endl;
            return false;
        }
        
        // Create runtime and execution context
        runtime = nvinfer1::createInferRuntime(gLogger);
        context = engine->createExecutionContext();
        
        // Setup bindings
        setup_bindings();
        
        // Cleanup
        delete parser;
        delete network;
        delete config;
        delete builder;
        
        return true;
    }
    
    bool serialize_engine(const std::string& filename) {
        if (!engine) return false;
        
        auto serialized = engine->serialize();
        std::ofstream file(filename, std::ios::binary);
        file.write(static_cast<const char*>(serialized->data()), serialized->size());
        
        delete serialized;
        return true;
    }
    
    bool deserialize_engine(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;
        
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        
        runtime = nvinfer1::createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(buffer.data(), size);
        context = engine->createExecutionContext();
        
        setup_bindings();
        return true;
    }
    
    bool infer(const std::vector<float*>& inputs, const std::vector<float*>& outputs) {
        if (!context) return false;
        
        // Copy inputs to GPU
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            if (engine->bindingIsInput(i)) {
                int input_idx = get_input_index(i);
                if (input_idx < inputs.size()) {
                    cudaMemcpyAsync(bindings[i], inputs[input_idx], binding_sizes[i],
                                  cudaMemcpyHostToDevice, stream);
                }
            }
        }
        
        // Execute inference
        bool success = context->enqueueV2(bindings.data(), stream, nullptr);
        if (!success) return false;
        
        // Copy outputs back to CPU
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            if (!engine->bindingIsInput(i)) {
                int output_idx = get_output_index(i);
                if (output_idx < outputs.size()) {
                    cudaMemcpyAsync(outputs[output_idx], bindings[i], binding_sizes[i],
                                  cudaMemcpyDeviceToHost, stream);
                }
            }
        }
        
        cudaStreamSynchronize(stream);
        return true;
    }
    
private:
    void setup_bindings() {
        int num_bindings = engine->getNbBindings();
        bindings.resize(num_bindings);
        binding_sizes.resize(num_bindings);
        
        for (int i = 0; i < num_bindings; ++i) {
            auto dims = engine->getBindingDimensions(i);
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                size *= dims.d[j];
            }
            size *= sizeof(float); // Assuming float precision
            
            binding_sizes[i] = size;
            cudaMalloc(&bindings[i], size);
        }
    }
    
    void cleanup() {
        for (void* binding : bindings) {
            if (binding) cudaFree(binding);
        }
        
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }
    
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) override {
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    } gLogger;
};
```

### INT8 Calibration

```cpp
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
private:
    int batch_size;
    int input_count;
    int batch_idx;
    
    std::vector<std::string> image_files;
    std::vector<char> calibration_cache;
    
    void* device_input;
    size_t input_size;
    
public:
    Int8Calibrator(int batch_size, const std::vector<std::string>& calibration_images)
        : batch_size(batch_size), batch_idx(0), image_files(calibration_images) {
        
        input_count = image_files.size();
        
        // Assuming input dimensions (batch_size, 3, 224, 224)
        input_size = batch_size * 3 * 224 * 224 * sizeof(float);
        cudaMalloc(&device_input, input_size);
    }
    
    ~Int8Calibrator() {
        if (device_input) cudaFree(device_input);
    }
    
    int getBatchSize() const override {
        return batch_size;
    }
    
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
        if (batch_idx + batch_size > input_count) {
            return false; // No more batches
        }
        
        // Load and preprocess batch of images
        std::vector<float> batch_data(batch_size * 3 * 224 * 224);
        
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx + i < input_count) {
                load_and_preprocess_image(image_files[batch_idx + i], 
                                        &batch_data[i * 3 * 224 * 224]);
            }
        }
        
        // Copy to GPU
        cudaMemcpy(device_input, batch_data.data(), input_size, cudaMemcpyHostToDevice);
        bindings[0] = device_input;
        
        batch_idx += batch_size;
        return true;
    }
    
    const void* readCalibrationCache(size_t& length) override {
        length = calibration_cache.size();
        return length ? calibration_cache.data() : nullptr;
    }
    
    void writeCalibrationCache(const void* cache, size_t length) override {
        calibration_cache.assign(static_cast<const char*>(cache),
                               static_cast<const char*>(cache) + length);
    }
    
private:
    void load_and_preprocess_image(const std::string& filename, float* output) {
        // Load image using OpenCV or similar
        cv::Mat image = cv::imread(filename);
        if (image.empty()) return;
        
        // Resize to 224x224
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(224, 224));
        
        // Convert to float and normalize
        resized.convertTo(resized, CV_32F, 1.0/255.0);
        
        // Convert BGR to RGB and rearrange to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);
        
        // Copy channels in RGB order
        memcpy(output, channels[2].data, 224 * 224 * sizeof(float)); // R
        memcpy(output + 224 * 224, channels[1].data, 224 * 224 * sizeof(float)); // G
        memcpy(output + 2 * 224 * 224, channels[0].data, 224 * 224 * sizeof(float)); // B
        
        // Apply ImageNet normalization
        float mean[3] = {0.485f, 0.456f, 0.406f};
        float std[3] = {0.229f, 0.224f, 0.225f};
        
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 224 * 224; ++i) {
                output[c * 224 * 224 + i] = (output[c * 224 * 224 + i] - mean[c]) / std[c];
            }
        }
    }
};
```

## NCCL for Multi-GPU Communication

### All-Reduce Operations

```cpp
#include <nccl.h>

class NCCLCommunicator {
private:
    ncclComm_t* comms;
    int num_gpus;
    cudaStream_t* streams;
    
public:
    NCCLCommunicator(int num_gpus) : num_gpus(num_gpus) {
        // Allocate communicators and streams
        comms = new ncclComm_t[num_gpus];
        streams = new cudaStream_t[num_gpus];
        
        // Initialize NCCL communicators
        int devices[num_gpus];
        for (int i = 0; i < num_gpus; ++i) {
            devices[i] = i;
        }
        
        ncclCommInitAll(comms, num_gpus, devices);
        
        // Create streams for each GPU
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            cudaStreamCreate(&streams[i]);
        }
    }
    
    ~NCCLCommunicator() {
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            ncclCommDestroy(comms[i]);
            cudaStreamDestroy(streams[i]);
        }
        delete[] comms;
        delete[] streams;
    }
    
    void all_reduce(float** data_ptrs, size_t count) {
        // Start group of NCCL operations
        ncclGroupStart();
        
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            ncclAllReduce(data_ptrs[i], data_ptrs[i], count, ncclFloat, ncclSum,
                         comms[i], streams[i]);
        }
        
        // End group
        ncclGroupEnd();
        
        // Synchronize all streams
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }
    }
    
    void all_gather(float** input_ptrs, float** output_ptrs, size_t count) {
        ncclGroupStart();
        
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            ncclAllGather(input_ptrs[i], output_ptrs[i], count, ncclFloat,
                         comms[i], streams[i]);
        }
        
        ncclGroupEnd();
        
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }
    }
    
    void broadcast(float* data, size_t count, int root_gpu) {
        ncclGroupStart();
        
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            ncclBcast(data, count, ncclFloat, root_gpu, comms[i], streams[i]);
        }
        
        ncclGroupEnd();
        
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }
    }
};

// Distributed training example
class DistributedTrainer {
private:
    NCCLCommunicator* nccl_comm;
    std::vector<ConvolutionLayer*> conv_layers;
    int num_gpus;
    int local_batch_size;
    
public:
    DistributedTrainer(int num_gpus, int total_batch_size) :
        num_gpus(num_gpus), local_batch_size(total_batch_size / num_gpus) {
        
        nccl_comm = new NCCLCommunicator(num_gpus);
        
        // Initialize layers on each GPU
        conv_layers.resize(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            // Initialize layer parameters...
            conv_layers[i] = new ConvolutionLayer(/* parameters */);
        }
    }
    
    void train_step(float** inputs, float** targets) {
        // Forward pass on each GPU
        std::vector<float*> outputs(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            conv_layers[i]->forward(inputs[i], outputs[i]);
        }
        
        // Compute gradients
        std::vector<float*> gradients(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            compute_gradients(outputs[i], targets[i], gradients[i]);
        }
        
        // All-reduce gradients
        nccl_comm->all_reduce(gradients.data(), get_gradient_count());
        
        // Average gradients
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            scale_gradients(gradients[i], 1.0f / num_gpus);
        }
        
        // Update parameters
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            conv_layers[i]->update_parameters(gradients[i]);
        }
    }
};
```

## Custom CUDA Kernels for ML

### Fused Operations

```cpp
// Fused convolution + batch norm + ReLU kernel
__global__ void fused_conv_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_size, int padding, int stride,
    float bn_epsilon
) {
    int batch_idx = blockIdx.x;
    int out_c = blockIdx.y;
    int out_h = threadIdx.y;
    int out_w = threadIdx.x;
    
    if (batch_idx >= batch_size || out_c >= out_channels || 
        out_h >= out_height || out_w >= out_width) return;
    
    float sum = 0.0f;
    
    // Convolution
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h = out_h * stride - padding + kh;
                int in_w = out_w * stride - padding + kw;
                
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    int input_idx = ((batch_idx * in_channels + in_c) * in_height + in_h) * in_width + in_w;
                    int weight_idx = ((out_c * in_channels + in_c) * kernel_size + kh) * kernel_size + kw;
                    
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_c];
    
    // Batch normalization
    float bn_out = (sum - bn_mean[out_c]) * rsqrtf(bn_var[out_c] + bn_epsilon);
    bn_out = bn_out * bn_scale[out_c] + bn_bias[out_c];
    
    // ReLU activation
    bn_out = fmaxf(0.0f, bn_out);
    
    int output_idx = ((batch_idx * out_channels + out_c) * out_height + out_h) * out_width + out_w;
    output[output_idx] = bn_out;
}

// Optimized matrix multiplication with tensor cores
__global__ void gemm_tensor_core_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    // Use WMMA API for tensor cores
    using namespace nvcuda;
    
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Bounds checking
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;
    
    // Initialize accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking for K dimension
        if (aCol < K && bRow < K) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store the output
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}
```

### Memory Layout Optimizations

```cpp
// Memory layout transformation kernels
__global__ void nchw_to_nhwc_kernel(const float* input, float* output,
                                   int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    
    if (idx >= total_elements) return;
    
    // Convert linear index to NCHW coordinates
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (C * H * W);
    
    // Calculate output index in NHWC format
    int out_idx = ((n * H + h) * W + w) * C + c;
    output[out_idx] = input[idx];
}

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[32][33]; // 33 to avoid bank conflicts
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Coalesced read from input
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Calculate transposed coordinates
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    // Coalesced write to output
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Optimized softmax kernel
__global__ void softmax_kernel(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* input_row = input + batch_idx * num_classes;
    float* output_row = output + batch_idx * num_classes;
    
    // Find maximum for numerical stability
    __shared__ float max_val;
    if (threadIdx.x == 0) {
        max_val = input_row[0];
        for (int i = 1; i < num_classes; ++i) {
            max_val = fmaxf(max_val, input_row[i]);
        }
    }
    __syncthreads();
    
    // Compute exponentials and sum
    __shared__ float sum;
    if (threadIdx.x == 0) {
        sum = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            sum += expf(input_row[i] - max_val);
        }
    }
    __syncthreads();
    
    // Compute softmax
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - max_val) / sum;
    }
}
```

## Performance Optimization and Profiling

```cpp
class MLPerformanceProfiler {
private:
    std::map<std::string, std::vector<float>> kernel_times;
    std::map<std::string, cudaEvent_t> start_events;
    std::map<std::string, cudaEvent_t> stop_events;
    
public:
    MLPerformanceProfiler() {
        // Pre-create events for common operations
        std::vector<std::string> operations = {
            "convolution", "batch_norm", "activation", "pooling", 
            "linear", "attention", "gradient_computation"
        };
        
        for (const auto& op : operations) {
            cudaEventCreate(&start_events[op]);
            cudaEventCreate(&stop_events[op]);
        }
    }
    
    void start_timer(const std::string& operation) {
        cudaEventRecord(start_events[operation]);
    }
    
    void end_timer(const std::string& operation) {
        cudaEventRecord(stop_events[operation]);
        cudaEventSynchronize(stop_events[operation]);
        
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_events[operation], stop_events[operation]);
        kernel_times[operation].push_back(elapsed_time);
    }
    
    void print_profile_report() {
        std::cout << "\n=== ML Performance Profile Report ===" << std::endl;
        for (const auto& entry : kernel_times) {
            const std::string& op = entry.first;
            const auto& times = entry.second;
            
            if (times.empty()) continue;
            
            float total_time = std::accumulate(times.begin(), times.end(), 0.0f);
            float avg_time = total_time / times.size();
            float min_time = *std::min_element(times.begin(), times.end());
            float max_time = *std::max_element(times.begin(), times.end());
            
            std::cout << op << ":" << std::endl;
            std::cout << "  Total: " << total_time << " ms" << std::endl;
            std::cout << "  Average: " << avg_time << " ms" << std::endl;
            std::cout << "  Min: " << min_time << " ms" << std::endl;
            std::cout << "  Max: " << max_time << " ms" << std::endl;
            std::cout << "  Calls: " << times.size() << std::endl;
        }
    }
    
    void analyze_memory_usage() {
        size_t free_bytes, total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        
        size_t used_bytes = total_bytes - free_bytes;
        
        std::cout << "\n=== GPU Memory Usage ===" << std::endl;
        std::cout << "Total: " << total_bytes / (1024*1024) << " MB" << std::endl;
        std::cout << "Used: " << used_bytes / (1024*1024) << " MB" << std::endl;
        std::cout << "Free: " << free_bytes / (1024*1024) << " MB" << std::endl;
        std::cout << "Utilization: " << (float)used_bytes / total_bytes * 100 << "%" << std::endl;
    }
};
```

## Exercises

1. **Custom Layer Implementation**: Implement a complete custom layer (e.g., Group Normalization) using cuDNN primitives.

2. **TensorRT Optimization**: Take a trained PyTorch/TensorFlow model, convert it to ONNX, and optimize it with TensorRT.

3. **Multi-GPU Training**: Implement data-parallel training using NCCL for gradient synchronization.

4. **Kernel Fusion**: Create fused kernels that combine multiple operations to reduce memory bandwidth.

5. **Mixed Precision Training**: Implement automatic mixed precision training with loss scaling.

## Key Takeaways

- cuDNN provides highly optimized primitives for deep learning operations
- TensorRT enables significant inference speedup through optimization techniques
- NCCL facilitates efficient multi-GPU communication for distributed training
- Custom CUDA kernels can provide additional optimization opportunities
- Memory layout and access patterns are critical for performance

## Next Steps

Proceed to [CUDA Programming Patterns](08_CUDA_Programming_Patterns.md) to learn about common parallel computing patterns and their efficient implementation in CUDA.
