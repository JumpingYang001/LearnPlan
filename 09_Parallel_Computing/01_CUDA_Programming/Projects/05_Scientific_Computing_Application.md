# Project 5: Scientific Computing Application

## Overview

Develop a comprehensive scientific computing application using CUDA to solve complex physics simulations. This project focuses on fluid dynamics simulation with real-time visualization, demonstrating advanced CUDA programming techniques for scientific computing.

## Project Goals

- Implement Navier-Stokes equations for fluid simulation
- Create real-time visualization with OpenGL interoperability
- Optimize for high-performance computing workloads
- Demonstrate domain-specific CUDA optimizations

## Physics Background

### Navier-Stokes Equations

The incompressible Navier-Stokes equations govern fluid motion:

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
∇·u = 0
```

Where:
- u: velocity field
- p: pressure
- ρ: density
- ν: kinematic viscosity
- f: external forces

## System Architecture

```cpp
// Core fluid simulation framework
class FluidSimulation {
private:
    struct SimulationGrid {
        int width, height, depth;
        float dx, dy, dz;  // Grid spacing
        float dt;          // Time step
        
        // Fluid properties
        float viscosity;
        float density;
        float diffusion_rate;
    };
    
    struct FluidState {
        // Velocity fields
        float4 *velocity_current;
        float4 *velocity_previous;
        
        // Pressure field
        float *pressure;
        float *pressure_temp;
        
        // Density/concentration field
        float *density_field;
        float *density_temp;
        
        // Boundary conditions
        uchar *boundary_mask;
    };
    
    SimulationGrid grid;
    FluidState state;
    
    // CUDA streams for overlapped computation
    cudaStream_t computation_stream;
    cudaStream_t visualization_stream;
    
    // OpenGL interop resources
    cudaGraphicsResource_t cuda_vbo_resource;
    cudaGraphicsResource_t cuda_texture_resource;
    
public:
    FluidSimulation(int width, int height, int depth);
    ~FluidSimulation();
    
    void initialize();
    void step_simulation();
    void render();
    void add_force(float x, float y, float fx, float fy);
    void add_density(float x, float y, float amount);
};
```

## Core CUDA Kernels

### 1. Advection Kernel

```cpp
// Semi-Lagrangian advection for velocity and density
__global__ void advect_kernel(float4* velocity_out, float4* velocity_in,
                            float4* velocity_field, float dt, float dx, float dy,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Current velocity
    float4 vel = velocity_field[idx];
    
    // Trace backward in time
    float prev_x = x - dt * vel.x / dx;
    float prev_y = y - dt * vel.y / dy;
    
    // Clamp to boundaries
    prev_x = fmaxf(0.5f, fminf(width - 0.5f, prev_x));
    prev_y = fmaxf(0.5f, fminf(height - 0.5f, prev_y));
    
    // Bilinear interpolation
    int i0 = (int)prev_x;
    int i1 = i0 + 1;
    int j0 = (int)prev_y;
    int j1 = j0 + 1;
    
    float s1 = prev_x - i0;
    float s0 = 1.0f - s1;
    float t1 = prev_y - j0;
    float t0 = 1.0f - t1;
    
    // Ensure indices are within bounds
    i0 = max(0, min(width - 1, i0));
    i1 = max(0, min(width - 1, i1));
    j0 = max(0, min(height - 1, j0));
    j1 = max(0, min(height - 1, j1));
    
    float4 val00 = velocity_in[j0 * width + i0];
    float4 val01 = velocity_in[j0 * width + i1];
    float4 val10 = velocity_in[j1 * width + i0];
    float4 val11 = velocity_in[j1 * width + i1];
    
    velocity_out[idx] = make_float4(
        s0 * (t0 * val00.x + t1 * val10.x) + s1 * (t0 * val01.x + t1 * val11.x),
        s0 * (t0 * val00.y + t1 * val10.y) + s1 * (t0 * val01.y + t1 * val11.y),
        s0 * (t0 * val00.z + t1 * val10.z) + s1 * (t0 * val01.z + t1 * val11.z),
        0.0f
    );
}

// Advection for scalar fields (density, temperature)
__global__ void advect_scalar_kernel(float* scalar_out, float* scalar_in,
                                   float4* velocity_field, float dt, float dx, float dy,
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float4 vel = velocity_field[idx];
    
    float prev_x = x - dt * vel.x / dx;
    float prev_y = y - dt * vel.y / dy;
    
    prev_x = fmaxf(0.5f, fminf(width - 0.5f, prev_x));
    prev_y = fmaxf(0.5f, fminf(height - 0.5f, prev_y));
    
    int i0 = (int)prev_x;
    int i1 = i0 + 1;
    int j0 = (int)prev_y;
    int j1 = j0 + 1;
    
    float s1 = prev_x - i0;
    float s0 = 1.0f - s1;
    float t1 = prev_y - j0;
    float t0 = 1.0f - t1;
    
    i0 = max(0, min(width - 1, i0));
    i1 = max(0, min(width - 1, i1));
    j0 = max(0, min(height - 1, j0));
    j1 = max(0, min(height - 1, j1));
    
    float val00 = scalar_in[j0 * width + i0];
    float val01 = scalar_in[j0 * width + i1];
    float val10 = scalar_in[j1 * width + i0];
    float val11 = scalar_in[j1 * width + i1];
    
    scalar_out[idx] = s0 * (t0 * val00 + t1 * val10) + s1 * (t0 * val01 + t1 * val11);
}
```

### 2. Diffusion Kernel (Jacobi Iteration)

```cpp
// Gauss-Seidel iteration for diffusion equation
__global__ void diffuse_kernel(float4* velocity_out, float4* velocity_in,
                             float viscosity, float dt, float dx, float dy,
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) return; // Skip boundaries
    
    int idx = y * width + x;
    
    float alpha = dt * viscosity / (dx * dx);
    float beta = 1.0f + 4.0f * alpha;
    
    float4 center = velocity_in[idx];
    float4 left = velocity_in[y * width + (x - 1)];
    float4 right = velocity_in[y * width + (x + 1)];
    float4 bottom = velocity_in[(y - 1) * width + x];
    float4 top = velocity_in[(y + 1) * width + x];
    
    velocity_out[idx] = make_float4(
        (center.x + alpha * (left.x + right.x + bottom.x + top.x)) / beta,
        (center.y + alpha * (left.y + right.y + bottom.y + top.y)) / beta,
        (center.z + alpha * (left.z + right.z + bottom.z + top.z)) / beta,
        0.0f
    );
}

// Red-black Gauss-Seidel for better convergence
__global__ void diffuse_red_black_kernel(float4* velocity, float4* velocity_temp,
                                        float viscosity, float dt, float dx, float dy,
                                        int width, int height, int phase) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) return;
    
    // Red-black checkerboard pattern
    if ((x + y) % 2 != phase) return;
    
    int idx = y * width + x;
    
    float alpha = dt * viscosity / (dx * dx);
    float beta = 1.0f + 4.0f * alpha;
    
    float4 center = velocity_temp[idx];
    float4 left = velocity[y * width + (x - 1)];
    float4 right = velocity[y * width + (x + 1)];
    float4 bottom = velocity[(y - 1) * width + x];
    float4 top = velocity[(y + 1) * width + x];
    
    velocity[idx] = make_float4(
        (center.x + alpha * (left.x + right.x + bottom.x + top.x)) / beta,
        (center.y + alpha * (left.y + right.y + bottom.y + top.y)) / beta,
        (center.z + alpha * (left.z + right.z + bottom.z + top.z)) / beta,
        0.0f
    );
}
```

### 3. Pressure Projection Kernels

```cpp
// Compute divergence of velocity field
__global__ void compute_divergence_kernel(float* divergence, float4* velocity,
                                        float dx, float dy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        divergence[y * width + x] = 0.0f;
        return;
    }
    
    int idx = y * width + x;
    
    float4 left = velocity[y * width + (x - 1)];
    float4 right = velocity[y * width + (x + 1)];
    float4 bottom = velocity[(y - 1) * width + x];
    float4 top = velocity[(y + 1) * width + x];
    
    float div_x = (right.x - left.x) / (2.0f * dx);
    float div_y = (top.y - bottom.y) / (2.0f * dy);
    
    divergence[idx] = -(div_x + div_y);
}

// Solve Poisson equation for pressure using Jacobi iteration
__global__ void pressure_jacobi_kernel(float* pressure_out, float* pressure_in,
                                     float* divergence, float dx, float dy,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        pressure_out[y * width + x] = 0.0f; // Dirichlet boundary conditions
        return;
    }
    
    int idx = y * width + x;
    
    float left = pressure_in[y * width + (x - 1)];
    float right = pressure_in[y * width + (x + 1)];
    float bottom = pressure_in[(y - 1) * width + x];
    float top = pressure_in[(y + 1) * width + x];
    
    float alpha = dx * dx; // Assuming dx = dy
    float beta = 4.0f;
    
    pressure_out[idx] = (alpha * divergence[idx] + left + right + bottom + top) / beta;
}

// Subtract pressure gradient from velocity
__global__ void subtract_pressure_gradient_kernel(float4* velocity, float* pressure,
                                                float dx, float dy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) return;
    
    int idx = y * width + x;
    
    float left = pressure[y * width + (x - 1)];
    float right = pressure[y * width + (x + 1)];
    float bottom = pressure[(y - 1) * width + x];
    float top = pressure[(y + 1) * width + x];
    
    float grad_x = (right - left) / (2.0f * dx);
    float grad_y = (top - bottom) / (2.0f * dy);
    
    float4 vel = velocity[idx];
    velocity[idx] = make_float4(vel.x - grad_x, vel.y - grad_y, vel.z, vel.w);
}
```

### 4. External Forces and Boundary Conditions

```cpp
// Apply external forces (mouse interaction, gravity, etc.)
__global__ void apply_forces_kernel(float4* velocity, float4* forces,
                                  float dt, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float4 vel = velocity[idx];
    float4 force = forces[idx];
    
    velocity[idx] = make_float4(
        vel.x + dt * force.x,
        vel.y + dt * force.y,
        vel.z + dt * force.z,
        vel.w
    );
    
    // Clear forces after application
    forces[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// Enforce boundary conditions
__global__ void enforce_boundaries_kernel(float4* velocity, uchar* boundary_mask,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Free slip boundary conditions at domain edges
    if (x == 0) {
        velocity[idx].x = 0.0f; // No flow through left boundary
    } else if (x == width - 1) {
        velocity[idx].x = 0.0f; // No flow through right boundary
    }
    
    if (y == 0) {
        velocity[idx].y = 0.0f; // No flow through bottom boundary
    } else if (y == height - 1) {
        velocity[idx].y = 0.0f; // No flow through top boundary
    }
    
    // Custom boundary conditions based on mask
    if (boundary_mask[idx] != 0) {
        velocity[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // No-slip condition
    }
}
```

## Simulation Engine Implementation

```cpp
class FluidSimulation {
private:
    SimulationGrid grid;
    FluidState state;
    
    // Simulation parameters
    float total_time;
    int iteration_count;
    
    // Performance monitoring
    cudaEvent_t start_event, stop_event;
    std::vector<float> frame_times;
    
public:
    FluidSimulation(int width, int height, int depth = 1) {
        grid.width = width;
        grid.height = height;
        grid.depth = depth;
        grid.dx = 1.0f / width;
        grid.dy = 1.0f / height;
        grid.dz = 1.0f / depth;
        grid.dt = 0.016f; // 60 FPS target
        grid.viscosity = 0.0001f;
        grid.density = 1.0f;
        grid.diffusion_rate = 0.0001f;
        
        total_time = 0.0f;
        iteration_count = 0;
        
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    void initialize() {
        size_t grid_size = grid.width * grid.height * grid.depth;
        
        // Allocate GPU memory
        cudaMalloc(&state.velocity_current, grid_size * sizeof(float4));
        cudaMalloc(&state.velocity_previous, grid_size * sizeof(float4));
        cudaMalloc(&state.pressure, grid_size * sizeof(float));
        cudaMalloc(&state.pressure_temp, grid_size * sizeof(float));
        cudaMalloc(&state.density_field, grid_size * sizeof(float));
        cudaMalloc(&state.density_temp, grid_size * sizeof(float));
        cudaMalloc(&state.boundary_mask, grid_size * sizeof(uchar));
        
        // Initialize fields
        cudaMemset(state.velocity_current, 0, grid_size * sizeof(float4));
        cudaMemset(state.velocity_previous, 0, grid_size * sizeof(float4));
        cudaMemset(state.pressure, 0, grid_size * sizeof(float));
        cudaMemset(state.pressure_temp, 0, grid_size * sizeof(float));
        cudaMemset(state.density_field, 0, grid_size * sizeof(float));
        cudaMemset(state.density_temp, 0, grid_size * sizeof(float));
        cudaMemset(state.boundary_mask, 0, grid_size * sizeof(uchar));
        
        // Create CUDA streams
        cudaStreamCreate(&computation_stream);
        cudaStreamCreate(&visualization_stream);
        
        std::cout << "Fluid simulation initialized: " << grid.width << "x" << grid.height << std::endl;
    }
    
    void step_simulation() {
        cudaEventRecord(start_event);
        
        dim3 block_size(16, 16);
        dim3 grid_size((grid.width + block_size.x - 1) / block_size.x,
                      (grid.height + block_size.y - 1) / block_size.y);
        
        // Simulation steps following Stam's "Stable Fluids" method
        
        // 1. Add forces
        apply_forces_kernel<<<grid_size, block_size, 0, computation_stream>>>(
            state.velocity_current, state.forces, grid.dt, grid.width, grid.height);
        
        // 2. Diffusion step
        std::swap(state.velocity_current, state.velocity_previous);
        for (int iter = 0; iter < 20; iter++) {
            diffuse_kernel<<<grid_size, block_size, 0, computation_stream>>>(
                state.velocity_current, state.velocity_previous,
                grid.viscosity, grid.dt, grid.dx, grid.dy, grid.width, grid.height);
            std::swap(state.velocity_current, state.velocity_previous);
        }
        
        // 3. Projection step (ensure incompressibility)
        project_velocity();
        
        // 4. Advection step
        std::swap(state.velocity_current, state.velocity_previous);
        advect_kernel<<<grid_size, block_size, 0, computation_stream>>>(
            state.velocity_current, state.velocity_previous, state.velocity_previous,
            grid.dt, grid.dx, grid.dy, grid.width, grid.height);
        
        // 5. Final projection
        project_velocity();
        
        // 6. Advect density
        std::swap(state.density_field, state.density_temp);
        advect_scalar_kernel<<<grid_size, block_size, 0, computation_stream>>>(
            state.density_field, state.density_temp, state.velocity_current,
            grid.dt, grid.dx, grid.dy, grid.width, grid.height);
        
        // 7. Enforce boundary conditions
        enforce_boundaries_kernel<<<grid_size, block_size, 0, computation_stream>>>(
            state.velocity_current, state.boundary_mask, grid.width, grid.height);
        
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        
        // Update performance metrics
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        frame_times.push_back(elapsed_time);
        
        total_time += grid.dt;
        iteration_count++;
        
        if (iteration_count % 60 == 0) {
            print_performance_stats();
        }
    }
    
private:
    void project_velocity() {
        dim3 block_size(16, 16);
        dim3 grid_size((grid.width + block_size.x - 1) / block_size.x,
                      (grid.height + block_size.y - 1) / block_size.y);
        
        // Compute divergence
        compute_divergence_kernel<<<grid_size, block_size, 0, computation_stream>>>(
            state.pressure_temp, state.velocity_current,
            grid.dx, grid.dy, grid.width, grid.height);
        
        // Solve Poisson equation for pressure (Jacobi iterations)
        cudaMemset(state.pressure, 0, grid.width * grid.height * sizeof(float));
        
        for (int iter = 0; iter < 40; iter++) {
            pressure_jacobi_kernel<<<grid_size, block_size, 0, computation_stream>>>(
                state.pressure, state.pressure_temp, state.pressure_temp,
                grid.dx, grid.dy, grid.width, grid.height);
            std::swap(state.pressure, state.pressure_temp);
        }
        
        // Subtract pressure gradient
        subtract_pressure_gradient_kernel<<<grid_size, block_size, 0, computation_stream>>>(
            state.velocity_current, state.pressure,
            grid.dx, grid.dy, grid.width, grid.height);
    }
    
    void print_performance_stats() {
        if (frame_times.empty()) return;
        
        float avg_time = std::accumulate(frame_times.begin(), frame_times.end(), 0.0f) / frame_times.size();
        float max_time = *std::max_element(frame_times.begin(), frame_times.end());
        float min_time = *std::min_element(frame_times.begin(), frame_times.end());
        
        std::cout << "Performance Stats (last 60 frames):" << std::endl;
        std::cout << "  Average: " << avg_time << "ms (" << 1000.0f / avg_time << " FPS)" << std::endl;
        std::cout << "  Min: " << min_time << "ms (" << 1000.0f / min_time << " FPS)" << std::endl;
        std::cout << "  Max: " << max_time << "ms (" << 1000.0f / max_time << " FPS)" << std::endl;
        std::cout << "  Simulation time: " << total_time << "s" << std::endl;
        
        frame_times.clear();
    }
};
```

## OpenGL Visualization

```cpp
// OpenGL-CUDA interoperability for real-time visualization
class FluidRenderer {
private:
    GLuint vertex_array, vertex_buffer;
    GLuint texture_id;
    GLuint shader_program;
    
    cudaGraphicsResource_t cuda_vbo_resource;
    cudaGraphicsResource_t cuda_texture_resource;
    
    int width, height;
    
public:
    FluidRenderer(int w, int h) : width(w), height(h) {}
    
    void initialize() {
        // Create OpenGL resources
        glGenVertexArrays(1, &vertex_array);
        glGenBuffers(1, &vertex_buffer);
        glGenTextures(1, &texture_id);
        
        // Setup quad for rendering
        float vertices[] = {
            -1.0f, -1.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f,
             1.0f,  1.0f, 1.0f, 1.0f,
            -1.0f,  1.0f, 0.0f, 1.0f
        };
        
        glBindVertexArray(vertex_array);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        // Create texture for fluid visualization
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        // Register OpenGL resources with CUDA
        cudaGraphicsGLRegisterImage(&cuda_texture_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
        
        // Load and compile shaders
        shader_program = create_shader_program();
    }
    
    void render_fluid(FluidSimulation& simulation) {
        // Map OpenGL texture to CUDA
        cudaArray_t texture_array;
        cudaGraphicsMapResources(1, &cuda_texture_resource, 0);
        cudaGraphicsSubResourceGetMappedArray(&texture_array, cuda_texture_resource, 0, 0);
        
        // Launch CUDA kernel to generate visualization data
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        generate_visualization_kernel<<<grid, block>>>(
            texture_array, simulation.get_velocity_field(), simulation.get_density_field(),
            width, height);
        
        cudaGraphicsUnmapResources(1, &cuda_texture_resource, 0);
        
        // Render with OpenGL
        glUseProgram(shader_program);
        glBindVertexArray(vertex_array);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indices);
    }
    
private:
    GLuint create_shader_program() {
        const char* vertex_shader_source = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            
            out vec2 TexCoord;
            
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        
        const char* fragment_shader_source = R"(
            #version 330 core
            out vec4 FragColor;
            
            in vec2 TexCoord;
            uniform sampler2D fluidTexture;
            
            void main() {
                vec4 fluid_data = texture(fluidTexture, TexCoord);
                
                // Visualize velocity as color
                float velocity_magnitude = length(fluid_data.xy);
                vec3 velocity_color = vec3(velocity_magnitude, 0.0, 1.0 - velocity_magnitude);
                
                // Visualize density
                float density = fluid_data.z;
                vec3 density_color = vec3(density, density, density);
                
                // Combine visualizations
                FragColor = vec4(mix(velocity_color, density_color, 0.5), 1.0);
            }
        )";
        
        // Compile and link shaders (implementation omitted for brevity)
        return compile_shader_program(vertex_shader_source, fragment_shader_source);
    }
};

// CUDA kernel for generating visualization data
__global__ void generate_visualization_kernel(cudaArray_t texture_array,
                                            float4* velocity_field,
                                            float* density_field,
                                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float4 velocity = velocity_field[idx];
    float density = density_field[idx];
    
    // Create visualization data
    float4 vis_data = make_float4(
        velocity.x,  // Red channel: velocity x
        velocity.y,  // Green channel: velocity y
        density,     // Blue channel: density
        1.0f         // Alpha channel
    );
    
    // Write to texture
    surf2Dwrite(vis_data, texture_array, x * sizeof(float4), y);
}
```

## Interactive Controls

```cpp
class FluidController {
private:
    FluidSimulation* simulation;
    bool mouse_pressed;
    float last_mouse_x, last_mouse_y;
    
public:
    FluidController(FluidSimulation* sim) : simulation(sim), mouse_pressed(false) {}
    
    void handle_mouse_input(float x, float y, bool pressed) {
        if (pressed && mouse_pressed) {
            // Calculate mouse velocity
            float dx = x - last_mouse_x;
            float dy = y - last_mouse_y;
            
            // Add force at mouse position
            simulation->add_force(x, y, dx * 100.0f, dy * 100.0f);
            simulation->add_density(x, y, 10.0f);
        }
        
        mouse_pressed = pressed;
        last_mouse_x = x;
        last_mouse_y = y;
    }
    
    void handle_keyboard_input(char key) {
        switch (key) {
            case 'r':
                simulation->reset();
                break;
            case '+':
                simulation->increase_viscosity();
                break;
            case '-':
                simulation->decrease_viscosity();
                break;
            case 's':
                simulation->save_state("fluid_state.dat");
                break;
            case 'l':
                simulation->load_state("fluid_state.dat");
                break;
        }
    }
};
```

## Performance Analysis and Optimization

```cpp
class FluidBenchmark {
public:
    void run_benchmarks() {
        std::vector<std::pair<int, int>> resolutions = {
            {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}
        };
        
        for (const auto& res : resolutions) {
            benchmark_resolution(res.first, res.second);
        }
        
        benchmark_different_algorithms();
        benchmark_memory_patterns();
    }
    
private:
    void benchmark_resolution(int width, int height) {
        FluidSimulation sim(width, height);
        sim.initialize();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 100; i++) {
            sim.step_simulation();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        float fps = 100000.0f / duration.count();
        float throughput = (float)(width * height) * fps / 1e6; // Megapixels per second
        
        std::cout << "Resolution " << width << "x" << height << ": " 
                  << fps << " FPS, " << throughput << " Mpixels/s" << std::endl;
    }
    
    void benchmark_different_algorithms() {
        // Compare Jacobi vs Gauss-Seidel
        // Compare different advection schemes
        // Compare different boundary condition implementations
    }
    
    void benchmark_memory_patterns() {
        // Test different memory layouts (AoS vs SoA)
        // Test texture memory vs global memory
        // Test shared memory optimizations
    }
};
```

## Usage and Integration

```cpp
// Main application
int main() {
    // Initialize OpenGL context (using GLFW, GLUT, or similar)
    init_opengl_context();
    
    // Create simulation
    FluidSimulation simulation(512, 512);
    simulation.initialize();
    
    // Create renderer
    FluidRenderer renderer(512, 512);
    renderer.initialize();
    
    // Create controller
    FluidController controller(&simulation);
    
    // Main loop
    while (!should_close()) {
        // Handle input
        handle_input(&controller);
        
        // Step simulation
        simulation.step_simulation();
        
        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        renderer.render_fluid(simulation);
        swap_buffers();
    }
    
    return 0;
}
```

## Performance Targets and Optimization Goals

- **Real-time Performance**: 60+ FPS for 512x512 grid
- **Memory Efficiency**: <4GB GPU memory for 1024x1024 simulation
- **Numerical Stability**: Stable simulation for CFL numbers up to 5.0
- **Visual Quality**: Smooth, realistic fluid motion with minimal artifacts

## Extensions and Advanced Features

1. **3D Fluid Simulation**: Extend to full 3D Navier-Stokes
2. **Multi-phase Flows**: Oil-water interfaces, surface tension
3. **Thermal Effects**: Heat transfer and buoyancy
4. **Particle Systems**: Lagrangian particle tracking
5. **Turbulence Modeling**: Large Eddy Simulation (LES)
6. **Adaptive Mesh Refinement**: Dynamic grid resolution
7. **Multi-GPU Scaling**: Domain decomposition for large simulations

## Key Learning Outcomes

- Advanced numerical methods implementation in CUDA
- OpenGL-CUDA interoperability for real-time visualization
- Memory optimization techniques for scientific computing
- Performance profiling and bottleneck analysis
- Parallel algorithm design for PDEs
- Real-time interactive simulation systems
