# PyTorch Architecture and Basics

*Duration: 2 weeks*

## PyTorch Overview

PyTorch is a deep learning framework developed by Facebook's AI Research lab (FAIR) that has become one of the most popular choices for machine learning research and production. Understanding its architecture is crucial for effective development.

### What Makes PyTorch Special?

#### Dynamic Computation Graph (Define-by-Run)

Unlike static graphs where you define the computation graph once and then execute it multiple times, PyTorch uses **dynamic computation graphs** that are built on-the-fly during execution.

```python
import torch
import torch.nn as nn

# Dynamic graph example
def dynamic_network(x, use_dropout=True):
    """
    The computation graph changes based on runtime conditions!
    """
    x = torch.relu(x)
    
    # Graph structure depends on runtime condition
    if use_dropout:
        x = torch.dropout(x, p=0.5, training=True)
    
    # Even the number of layers can be dynamic
    for i in range(torch.randint(1, 4, (1,)).item()):
        x = torch.relu(torch.linear(x, torch.randn(x.size(-1), x.size(-1))))
    
    return x

# Each call can have a different computation graph!
x = torch.randn(10, 20)
output1 = dynamic_network(x, use_dropout=True)   # Graph A
output2 = dynamic_network(x, use_dropout=False)  # Graph B (different!)
```

**Static vs Dynamic Graph Comparison:**

```
Static Graph (TensorFlow 1.x style):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Define Graph  │ -> │  Compile Graph  │ -> │  Execute Graph  │
│   (Build once)  │    │     (Once)      │    │   (Multiple)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘

Dynamic Graph (PyTorch style):
┌─────────────────┐
│ Build & Execute │  <- Each forward pass builds a new graph
│   Simultaneously│     allowing for runtime flexibility
└─────────────────┘
```

#### Eager Execution by Default

PyTorch executes operations immediately as they are called, making debugging intuitive.

```python
import torch

# Immediate execution - you can inspect values at any point
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"Initial x: {x}")  # Values available immediately

y = x ** 2
print(f"After squaring: {y}")  # Can inspect intermediate results

z = y.sum()
print(f"After sum: {z}")

# Backward pass
z.backward()
print(f"Gradients: {x.grad}")  # Gradients computed and available

# You can even modify the computation mid-execution
if z > 10:
    w = z * 2
else:
    w = z + 1
print(f"Final result: {w}")
```

#### PyTorch Ecosystem

PyTorch has a rich ecosystem of libraries and tools:

```python
# Core PyTorch ecosystem components
"""
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch Ecosystem                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   PyTorch   │  │ TorchVision │  │ TorchAudio  │        │
│  │    Core     │  │   (Vision)  │  │   (Audio)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  TorchText  │  │  Lightning  │  │   Ignite    │        │
│  │    (NLP)    │  │ (Training)  │  │ (Training)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Captum    │  │   PyTorch   │  │  TorchServe │        │
│  │(Interpret.) │  │  Geometric  │  │  (Serving)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
"""

# Example: Using multiple ecosystem components
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# TorchVision for computer vision
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset using TorchVision
from torchvision.datasets import MNIST
dataset = MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# PyTorch Lightning for simplified training (optional)
import pytorch_lightning as pl

class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(28*28, 10)
    
    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
```

#### Comparison with TensorFlow

| Aspect | PyTorch | TensorFlow 2.x |
|--------|---------|----------------|
| **Graph Type** | Dynamic (Define-by-Run) | Eager + Static (Graph mode) |
| **Learning Curve** | Steeper initially, more Pythonic | Gentler, more abstractions |
| **Debugging** | Standard Python debugging | TensorFlow debugger tools |
| **Deployment** | TorchScript, ONNX | TensorFlow Serving, TF Lite |
| **Research** | Preferred in academia | Strong in both research & production |
| **Production** | Growing rapidly | Mature production ecosystem |
| **Memory Usage** | Higher due to dynamic graphs | More memory efficient |
| **Performance** | Competitive, improving | Generally faster |

```python
# PyTorch style (more explicit)
import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

model = PyTorchModel()
x = torch.randn(32, 10)
output = model(x)

# TensorFlow style (more functional)
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(10,))
])

x = tf.random.normal((32, 10))
output = model(x)
```

## Core Components

Understanding PyTorch's core components is essential for building effective deep learning models. Let's explore each component in detail with practical examples.

### Tensor Operations

**Tensors** are the fundamental data structure in PyTorch - multi-dimensional arrays that can run on GPUs and track gradients.

#### Tensor Creation and Basic Operations

```python
import torch
import numpy as np

# Various ways to create tensors
print("=== Tensor Creation ===")

# From Python lists
tensor_from_list = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"From list:\n{tensor_from_list}")

# Initialized tensors
zeros_tensor = torch.zeros(2, 3)
ones_tensor = torch.ones(2, 3)
random_tensor = torch.randn(2, 3)  # Normal distribution
uniform_tensor = torch.rand(2, 3)  # Uniform [0,1)

print(f"\nZeros:\n{zeros_tensor}")
print(f"Ones:\n{ones_tensor}")
print(f"Random normal:\n{random_tensor}")

# From NumPy arrays (shares memory!)
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"\nFrom NumPy:\n{tensor_from_numpy}")

# Device placement
if torch.cuda.is_available():
    gpu_tensor = torch.ones(2, 3).cuda()
    print(f"GPU tensor device: {gpu_tensor.device}")

# Tensor properties
print(f"\nTensor shape: {random_tensor.shape}")
print(f"Tensor dtype: {random_tensor.dtype}")
print(f"Tensor device: {random_tensor.device}")
print(f"Requires grad: {random_tensor.requires_grad}")
```

#### Advanced Tensor Operations

```python
# Mathematical operations
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

print("=== Mathematical Operations ===")

# Element-wise operations
addition = a + b
multiplication = a * b
division = a / b

print(f"Addition:\n{addition}")
print(f"Element-wise multiplication:\n{multiplication}")

# Matrix operations
matrix_mult = torch.mm(a, b)  # Matrix multiplication
print(f"Matrix multiplication:\n{matrix_mult}")

# Broadcasting (automatic size expansion)
c = torch.tensor([1., 2.])  # Shape: (2,)
broadcasted = a + c  # c is broadcast to (2, 2)
print(f"Broadcasting result:\n{broadcasted}")

# Reduction operations
sum_all = a.sum()
sum_dim0 = a.sum(dim=0)  # Sum along rows
sum_dim1 = a.sum(dim=1)  # Sum along columns

print(f"Sum all: {sum_all}")
print(f"Sum dim 0: {sum_dim0}")
print(f"Sum dim 1: {sum_dim1}")

# Reshaping and indexing
reshaped = a.view(1, 4)  # Reshape to 1x4
flattened = a.flatten()  # Flatten to 1D

print(f"Original shape: {a.shape}")
print(f"Reshaped: {reshaped.shape}")
print(f"Flattened: {flattened.shape}")

# Advanced indexing
print(f"First row: {a[0]}")
print(f"Last column: {a[:, -1]}")
print(f"Top-left 2x2: {a[:2, :2]}")

# Boolean indexing
mask = a > 2
print(f"Elements > 2: {a[mask]}")
```

### Autograd System

The **automatic differentiation (autograd)** system is PyTorch's engine for computing gradients automatically.

#### How Autograd Works

```python
import torch

print("=== Autograd Fundamentals ===")

# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Define computation
z = x**2 + y**3
loss = z.sum()

print(f"x = {x}")
print(f"y = {y}")
print(f"z = x² + y³ = {z}")
print(f"loss = {loss}")

# Compute gradients
loss.backward()

print(f"\n∂loss/∂x = ∂(x² + y³)/∂x = 2x = {x.grad}")
print(f"∂loss/∂y = ∂(x² + y³)/∂y = 3y² = {y.grad}")

# Verify manual calculation
print(f"Manual ∂loss/∂x = 2 * {x.item()} = {2 * x.item()}")
print(f"Manual ∂loss/∂y = 3 * {y.item()}² = {3 * y.item()**2}")
```

#### Computational Graph Visualization

```python
# Understanding the computational graph
import torch
import torch.nn as nn

print("=== Computational Graph Example ===")

# Create a more complex computation
x = torch.tensor([1.0], requires_grad=True)
w1 = torch.tensor([2.0], requires_grad=True)
w2 = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# Forward pass: y = w2 * tanh(w1 * x + b)
h = w1 * x + b
h_tanh = torch.tanh(h)
y = w2 * h_tanh

print(f"x = {x.item()}")
print(f"h = w1 * x + b = {w1.item()} * {x.item()} + {b.item()} = {h.item()}")
print(f"h_tanh = tanh({h.item()}) = {h_tanh.item():.4f}")
print(f"y = w2 * h_tanh = {w2.item()} * {h_tanh.item():.4f} = {y.item():.4f}")

# Backward pass
y.backward()

print(f"\nGradients:")
print(f"∂y/∂x = {x.grad.item():.4f}")
print(f"∂y/∂w1 = {w1.grad.item():.4f}")
print(f"∂y/∂w2 = {w2.grad.item():.4f}")
print(f"∂y/∂b = {b.grad.item():.4f}")

"""
Computational Graph:
    x ──┐
        ├─> h = w1*x + b ──> h_tanh = tanh(h) ──┐
    w1 ─┘                                        ├─> y = w2 * h_tanh
    b ──────────────────────────────────────────┘
    w2 ─────────────────────────────────────────┘
"""
```

#### Gradient Control and Advanced Features

```python
# Gradient control techniques
print("=== Advanced Autograd Features ===")

# 1. Detaching from computation graph
x = torch.tensor([1.0], requires_grad=True)
y = x**2

# Stop gradient flow
z1 = y.detach() * 3  # z1 won't receive gradients
z2 = y * 3          # z2 will receive gradients

loss1 = z1.sum()
loss2 = z2.sum()

# Only loss2.backward() will compute gradients for x
loss2.backward()
print(f"Gradient after detach: {x.grad}")

# 2. Gradient accumulation
x.grad.zero_()  # Reset gradients

for i in range(3):
    y = x**2 * (i + 1)
    y.backward(retain_graph=True)  # Keep graph for next iteration
    print(f"Iteration {i+1}, accumulated gradient: {x.grad.item()}")

# 3. Higher-order gradients
x.grad.zero_()
y = x**3
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]  # First derivative
grad2 = torch.autograd.grad(grad1, x)[0]  # Second derivative

print(f"\nFirst derivative (3x²): {grad1.item()}")
print(f"Second derivative (6x): {grad2.item()}")

# 4. No-grad context for inference
with torch.no_grad():
    y = x**2 * 5  # No gradients computed
    print(f"No grad computation: {y.requires_grad}")
```

### Optimization Algorithms

PyTorch provides various optimization algorithms through `torch.optim`.

#### Common Optimizers Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Simple function to optimize: f(x) = (x - 2)² + 1
def objective_function(x):
    return (x - 2)**2 + 1

def optimize_with_different_optimizers():
    print("=== Optimizer Comparison ===")
    
    # Starting point
    start_x = torch.tensor([5.0], requires_grad=True)
    
    optimizers = {
        'SGD': optim.SGD([start_x.clone().detach().requires_grad_()], lr=0.1),
        'Adam': optim.Adam([start_x.clone().detach().requires_grad_()], lr=0.1),
        'RMSprop': optim.RMSprop([start_x.clone().detach().requires_grad_()], lr=0.1),
        'Adagrad': optim.Adagrad([start_x.clone().detach().requires_grad_()], lr=0.1)
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        x = optimizer.param_groups[0]['params'][0]
        history = []
        
        for step in range(50):
            optimizer.zero_grad()
            loss = objective_function(x)
            loss.backward()
            optimizer.step()
            
            history.append(x.item())
            
            if step % 10 == 0:
                print(f"{name} Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")
        
        results[name] = history
        print(f"{name} final: x = {x.item():.4f} (target: 2.0)\n")
    
    return results

# Run optimization comparison
results = optimize_with_different_optimizers()
```

#### Practical Optimizer Usage

```python
# Real neural network optimization example
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model setup
model = SimpleNet(input_size=10, hidden_size=20, output_size=1)
criterion = nn.MSELoss()

# Different optimizer configurations
print("=== Optimizer Configurations ===")

# 1. Basic SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

# 2. SGD with momentum
optimizer_sgd_momentum = optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9,
    weight_decay=1e-4  # L2 regularization
)

# 3. Adam with custom parameters
optimizer_adam = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),  # Beta1, Beta2
    eps=1e-8,
    weight_decay=1e-4
)

# 4. Learning rate scheduling
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop example
def train_step(model, optimizer, scheduler, x, y):
    model.train()
    optimizer.zero_grad()
    
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    # Gradient clipping (optional)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if scheduler:
        scheduler.step()
    
    return loss.item()

# Example usage
x_sample = torch.randn(32, 10)  # Batch of 32, input size 10
y_sample = torch.randn(32, 1)   # Target values

loss = train_step(model, optimizer, scheduler, x_sample, y_sample)
print(f"Training loss: {loss:.4f}")
print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
```

### Neural Network Modules

The `torch.nn` module provides building blocks for neural networks.

#### Basic Neural Network Components

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Neural Network Modules ===")

# 1. Linear layers
linear_layer = nn.Linear(in_features=10, out_features=5)
print(f"Linear layer input size: {linear_layer.in_features}")
print(f"Linear layer output size: {linear_layer.out_features}")
print(f"Weight shape: {linear_layer.weight.shape}")
print(f"Bias shape: {linear_layer.bias.shape}")

# 2. Activation functions
activation_functions = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'Leaky ReLU': nn.LeakyReLU(0.1),
    'Swish': nn.SiLU(),  # Also known as Swish
    'GELU': nn.GELU()
}

x = torch.tensor([-2., -1., 0., 1., 2.])
print(f"\nInput: {x}")
for name, activation in activation_functions.items():
    output = activation(x)
    print(f"{name}: {output}")

# 3. Convolutional layers
conv2d = nn.Conv2d(
    in_channels=3,   # RGB input
    out_channels=16, # 16 filters
    kernel_size=3,   # 3x3 kernel
    stride=1,
    padding=1
)

# Example input: batch_size=1, channels=3, height=32, width=32
input_image = torch.randn(1, 3, 32, 32)
conv_output = conv2d(input_image)
print(f"\nConv2d input shape: {input_image.shape}")
print(f"Conv2d output shape: {conv_output.shape}")

# 4. Pooling layers
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

maxpool_output = maxpool(conv_output)
avgpool_output = avgpool(conv_output)

print(f"MaxPool output shape: {maxpool_output.shape}")
print(f"AvgPool output shape: {avgpool_output.shape}")

# 5. Normalization layers
batch_norm = nn.BatchNorm2d(16)  # 16 channels
layer_norm = nn.LayerNorm([16, 16, 16])  # Normalize over last 3 dimensions

bn_output = batch_norm(conv_output)
print(f"BatchNorm output shape: {bn_output.shape}")

# 6. Dropout for regularization
dropout = nn.Dropout(p=0.5)
dropout_output = dropout(conv_output)
print(f"Dropout output shape: {dropout_output.shape}")
```

#### Building Custom Neural Networks

```python
# Complete neural network examples
class CNNClassifier(nn.Module):
    """Convolutional Neural Network for image classification"""
    
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assuming 32x32 input
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with batch norm and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class RNNClassifier(nn.Module):
    """Recurrent Neural Network for sequence classification"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super(RNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM output
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Classification
        output = self.fc(self.dropout(last_hidden))
        return output

# Model instantiation and usage
print("=== Custom Neural Networks ===")

# CNN example
cnn_model = CNNClassifier(num_classes=10)
sample_image = torch.randn(4, 3, 32, 32)  # Batch of 4 images
cnn_output = cnn_model(sample_image)
print(f"CNN input shape: {sample_image.shape}")
print(f"CNN output shape: {cnn_output.shape}")

# RNN example
rnn_model = RNNClassifier(
    vocab_size=1000, 
    embed_dim=128, 
    hidden_dim=256, 
    num_classes=5
)
sample_sequence = torch.randint(0, 1000, (4, 20))  # Batch of 4 sequences, length 20
rnn_output = rnn_model(sample_sequence)
print(f"RNN input shape: {sample_sequence.shape}")
print(f"RNN output shape: {rnn_output.shape}")

# Model summary function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nCNN trainable parameters: {count_parameters(cnn_model):,}")
print(f"RNN trainable parameters: {count_parameters(rnn_model):,}")
```

## PyTorch C++ API (LibTorch)

LibTorch is the C++ frontend for PyTorch, enabling deployment of PyTorch models in production C++ applications. This is crucial for performance-critical applications and integration with existing C++ codebases.

### C++ Frontend Architecture

LibTorch follows a similar design philosophy to PyTorch but adapted for C++:

```
LibTorch Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   torch::   │  │   torch::   │  │   torch::   │        │
│  │     nn      │  │    optim    │  │    jit      │        │
│  │  (Modules)  │  │(Optimizers)│  │  (Script)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Core Library (ATen)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Tensor    │  │   Autograd  │  │   Backend   │        │
│  │ Operations  │  │   Engine    │  │   (CPU/GPU) │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Building and Linking

#### CMake Configuration

```cmake
# CMakeLists.txt for LibTorch project
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch_cpp_example)

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create executable
add_executable(pytorch_app main.cpp)
target_link_libraries(pytorch_app "${TORCH_LIBRARIES}")
set_property(TARGET pytorch_app PROPERTY CXX_STANDARD 14)

# Copy DLLs on Windows
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pytorch_app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:pytorch_app>)
endif (MSVC)
```

#### Installation and Setup

```bash
# Download LibTorch (Linux/Mac)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# Set environment
export CMAKE_PREFIX_PATH=/path/to/libtorch:$CMAKE_PREFIX_PATH

# Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

### Tensor Operations in C++

LibTorch tensors work similarly to PyTorch tensors but with C++ syntax:

```cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>

void tensor_operations_demo() {
    std::cout << "=== LibTorch Tensor Operations ===" << std::endl;
    
    // Tensor creation
    torch::Tensor ones = torch::ones({2, 3});
    torch::Tensor zeros = torch::zeros({2, 3});
    torch::Tensor random = torch::randn({2, 3});
    
    std::cout << "Ones tensor:\n" << ones << std::endl;
    std::cout << "Random tensor:\n" << random << std::endl;
    
    // From C++ data
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    torch::Tensor from_vector = torch::from_blob(
        data.data(), 
        {2, 3}, 
        torch::kFloat
    ).clone(); // Clone to own the data
    
    std::cout << "From vector:\n" << from_vector << std::endl;
    
    // Mathematical operations
    torch::Tensor a = torch::tensor({{1, 2}, {3, 4}}, torch::kFloat);
    torch::Tensor b = torch::tensor({{5, 6}, {7, 8}}, torch::kFloat);
    
    torch::Tensor addition = a + b;
    torch::Tensor matrix_mult = torch::mm(a, b);
    
    std::cout << "Matrix A:\n" << a << std::endl;
    std::cout << "Matrix B:\n" << b << std::endl;
    std::cout << "A + B:\n" << addition << std::endl;
    std::cout << "A * B:\n" << matrix_mult << std::endl;
    
    // GPU operations (if available)
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Moving to GPU..." << std::endl;
        torch::Device device(torch::kCUDA, 0);
        torch::Tensor gpu_tensor = a.to(device);
        torch::Tensor gpu_result = torch::mm(gpu_tensor, b.to(device));
        
        std::cout << "GPU computation result:\n" << gpu_result.cpu() << std::endl;
    }
    
    // Autograd in C++
    torch::Tensor x = torch::tensor({2.0}, torch::requires_grad(true));
    torch::Tensor y = torch::tensor({3.0}, torch::requires_grad(true));
    
    torch::Tensor z = x * x + y * y * y;
    z.backward();
    
    std::cout << "x gradient: " << x.grad() << std::endl;
    std::cout << "y gradient: " << y.grad() << std::endl;
}
```

### Model Loading and Inference

#### Loading TorchScript Models

```cpp
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>

class ModelInference {
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    
public:
    ModelInference(const std::string& model_path, bool use_gpu = false) 
        : device_(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        
        try {
            // Load the model
            model_ = torch::jit::load(model_path);
            model_.to(device_);
            model_.eval();
            
            std::cout << "Model loaded successfully on " 
                      << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
                      
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }
    
    torch::Tensor predict(const torch::Tensor& input) {
        // Ensure input is on the correct device
        torch::Tensor input_device = input.to(device_);
        
        // Disable gradient computation for inference
        torch::NoGradGuard no_grad;
        
        // Create inputs vector
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_device);
        
        // Forward pass
        at::Tensor output = model_.forward(inputs).toTensor();
        
        return output;
    }
    
    std::vector<torch::Tensor> predict_batch(const std::vector<torch::Tensor>& inputs) {
        std::vector<torch::Tensor> outputs;
        outputs.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            outputs.push_back(predict(input));
        }
        
        return outputs;
    }
};

// Example usage
void inference_example() {
    std::cout << "=== Model Inference Example ===" << std::endl;
    
    try {
        // Load model
        ModelInference inference("model.pt", true); // Use GPU if available
        
        // Prepare input data
        torch::Tensor input = torch::randn({1, 3, 224, 224}); // Batch size 1, RGB, 224x224
        
        // Perform inference
        auto start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor output = inference.predict(input);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Input shape: " << input.sizes() << std::endl;
        std::cout << "Output shape: " << output.sizes() << std::endl;
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        
        // Get predictions (assuming classification)
        torch::Tensor probabilities = torch::softmax(output, 1);
        torch::Tensor top_class = torch::argmax(probabilities, 1);
        
        std::cout << "Predicted class: " << top_class.item<int>() << std::endl;
        std::cout << "Confidence: " << torch::max(probabilities).item<float>() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
    }
}
```

### JIT Compilation

TorchScript allows you to serialize PyTorch models for production use in C++:

#### Creating TorchScript Models (Python side)

```python
import torch
import torch.nn as nn

# Method 1: Scripting
class ScriptedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return torch.relu(self.linear(x))

model = ScriptedModel()
scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pt")

# Method 2: Tracing
class TracedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TracedModel()
example_input = torch.randn(1, 3, 32, 32)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("traced_model.pt")

# Method 3: Hybrid approach
@torch.jit.script
def complex_function(x):
    if x.dim() > 1:
        return x.mean(dim=1)
    else:
        return x

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.linear(x)
        return complex_function(x)

model = HybridModel()
scripted_hybrid = torch.jit.script(model)
scripted_hybrid.save("hybrid_model.pt")
```

#### Using JIT Models in C++

```cpp
#include <torch/script.h>
#include <memory>

class JITModelManager {
private:
    std::unordered_map<std::string, torch::jit::script::Module> models_;
    
public:
    void load_model(const std::string& name, const std::string& path) {
        try {
            models_[name] = torch::jit::load(path);
            models_[name].eval();
            std::cout << "Loaded model: " << name << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load model " << name << ": " << e.what() << std::endl;
        }
    }
    
    torch::Tensor infer(const std::string& model_name, const torch::Tensor& input) {
        auto it = models_.find(model_name);
        if (it == models_.end()) {
            throw std::runtime_error("Model not found: " + model_name);
        }
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        return it->second.forward(inputs).toTensor();
    }
    
    // Batch inference with performance optimization
    std::vector<torch::Tensor> batch_infer(
        const std::string& model_name, 
        const std::vector<torch::Tensor>& inputs,
        size_t batch_size = 32) {
        
        std::vector<torch::Tensor> results;
        results.reserve(inputs.size());
        
        // Process in batches for efficiency
        for (size_t i = 0; i < inputs.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, inputs.size());
            
            // Create batch tensor
            std::vector<torch::Tensor> batch_inputs(
                inputs.begin() + i, 
                inputs.begin() + end
            );
            torch::Tensor batch_tensor = torch::stack(batch_inputs);
            
            // Infer on batch
            torch::Tensor batch_output = infer(model_name, batch_tensor);
            
            // Split results
            for (int j = 0; j < batch_output.size(0); ++j) {
                results.push_back(batch_output[j]);
            }
        }
        
        return results;
    }
};

// Performance benchmarking
void benchmark_inference() {
    std::cout << "=== Performance Benchmark ===" << std::endl;
    
    JITModelManager manager;
    manager.load_model("classifier", "traced_model.pt");
    
    // Generate test data
    std::vector<torch::Tensor> test_inputs;
    for (int i = 0; i < 1000; ++i) {
        test_inputs.push_back(torch::randn({3, 32, 32}));
    }
    
    // Benchmark single inference
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& input : test_inputs) {
        manager.infer("classifier", input.unsqueeze(0));
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto single_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Benchmark batch inference
    start = std::chrono::high_resolution_clock::now();
    auto batch_results = manager.batch_infer("classifier", test_inputs, 32);
    end = std::chrono::high_resolution_clock::now();
    
    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Single inference time: " << single_duration.count() << " ms" << std::endl;
    std::cout << "Batch inference time: " << batch_duration.count() << " ms" << std::endl;
    std::cout << "Speedup: " << static_cast<float>(single_duration.count()) / batch_duration.count() 
              << "x" << std::endl;
}
```

#### Complete C++ Application Example

```cpp
// main.cpp - Complete LibTorch application
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <chrono>

int main() {
    std::cout << "LibTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA Available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
    
    try {
        // Run all examples
        tensor_operations_demo();
        inference_example();
        benchmark_inference();
        
        std::cout << "All examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
```

### Integration Best Practices

1. **Memory Management**: Use RAII and smart pointers
2. **Error Handling**: Always wrap LibTorch calls in try-catch blocks
3. **Device Management**: Explicitly manage CPU/GPU placement
4. **Performance**: Use batch processing and async operations when possible
5. **Thread Safety**: LibTorch operations are generally thread-safe, but be careful with shared state

## PyTorch Model Representation

Understanding how PyTorch represents and serializes models is crucial for deployment, sharing, and production use. PyTorch offers multiple model representation formats, each with specific use cases.

### Model Representation Formats Overview

```
PyTorch Model Formats:
┌─────────────────────────────────────────────────────────────┐
│                   Development Phase                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Python    │  │   Pickle    │  │  State Dict │        │
│  │    Model    │  │    (.pkl)   │  │   (.pth)    │        │
│  │ (Runtime)   │  │  (Full obj) │  │(Params only)│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                   Production Phase                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ TorchScript │  │    ONNX     │  │  TensorRT   │        │
│  │   (.pt)     │  │   (.onnx)   │  │   (.plan)   │        │
│  │(Cross-lang) │  │(Framework-  │  │(NVIDIA GPU)│        │
│  │             │  │ agnostic)   │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### TorchScript

**TorchScript** is PyTorch's intermediate representation that allows you to run models independently of Python.

#### Creating TorchScript Models

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Example model for demonstration
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

print("=== TorchScript Creation Methods ===")

# Method 1: torch.jit.script (Static Analysis)
model = ConvNet()
model.eval()

try:
    scripted_model = torch.jit.script(model)
    print("✓ Scripting successful")
    
    # Save scripted model
    scripted_model.save("scripted_convnet.pt")
    print("✓ Scripted model saved")
    
except Exception as e:
    print(f"✗ Scripting failed: {e}")

# Method 2: torch.jit.trace (Dynamic Analysis)
sample_input = torch.randn(1, 3, 32, 32)

try:
    traced_model = torch.jit.trace(model, sample_input)
    print("✓ Tracing successful")
    
    # Save traced model
    traced_model.save("traced_convnet.pt")
    print("✓ Traced model saved")
    
except Exception as e:
    print(f"✗ Tracing failed: {e}")

# Method 3: Hybrid approach for complex models
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # This control flow can be scripted
        x = F.relu(self.bn(self.conv(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(batch_size, -1)
        
        # Conditional logic that TorchScript can handle
        if self.training:
            x = F.dropout(x, p=0.5)
        
        return self.fc(x)

hybrid_model = HybridModel()
hybrid_model.eval()

scripted_hybrid = torch.jit.script(hybrid_model)
scripted_hybrid.save("hybrid_model.pt")
print("✓ Hybrid model saved")
```

#### TorchScript Limitations and Solutions

```python
# Common TorchScript limitations and workarounds

# 1. Python lists and dictionaries
class ProblematicModel(nn.Module):
    def forward(self, x):
        # This won't work in TorchScript
        results = []
        for i in range(x.size(0)):
            results.append(x[i] * 2)
        return results

# Solution: Use tensor operations
class FixedModel(nn.Module):
    def forward(self, x):
        # Use tensor operations instead
        return x * 2

# 2. Complex control flow
class ComplexControlFlow(nn.Module):
    def forward(self, x):
        # TorchScript has limited support for complex loops
        output = x
        for i in range(10):
            if torch.rand(1).item() > 0.5:  # Non-deterministic - problematic
                output = output * 2
            else:
                output = output + 1
        return output

# Solution: Make control flow deterministic
class DeterministicControlFlow(nn.Module):
    def forward(self, x):
        # Use tensor-based conditions
        condition = torch.randint(0, 2, (1,)).bool()
        output = torch.where(condition, x * 2, x + 1)
        return output

# 3. External library calls
class ExternalLibModel(nn.Module):
    def forward(self, x):
        # This won't work - NumPy not supported
        # import numpy as np
        # return torch.from_numpy(np.array(x.cpu()))
        pass

# Solution: Use pure PyTorch operations
class PurePyTorchModel(nn.Module):
    def forward(self, x):
        # Equivalent PyTorch operation
        return x.clone()

print("=== TorchScript Best Practices ===")
print("✓ Use tensor operations instead of Python loops")
print("✓ Avoid external library dependencies") 
print("✓ Keep control flow simple and deterministic")
print("✓ Test scripted models thoroughly")
```

### Model Saving and Loading

PyTorch provides multiple ways to save and load models, each suited for different scenarios.

#### Complete Model Saving Strategies

```python
import torch
import torch.nn as nn
import os
from pathlib import Path

# Create example model and data
model = ConvNet(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Simulate training state
model.train()
sample_input = torch.randn(4, 3, 32, 32)
sample_target = torch.randint(0, 10, (4,))

output = model(sample_input)
loss = nn.CrossEntropyLoss()(output, sample_target)
loss.backward()
optimizer.step()

print("=== Model Saving Strategies ===")

# Strategy 1: Save entire model (NOT RECOMMENDED for production)
torch.save(model, 'complete_model.pth')
print("✓ Complete model saved (development only)")

# Strategy 2: Save state dictionary (RECOMMENDED)
torch.save({
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss.item(),
    'model_config': {
        'num_classes': 10,
        'architecture': 'ConvNet'
    }
}, 'model_checkpoint.pth')
print("✓ Model checkpoint saved (recommended)")

# Strategy 3: Save only model weights for inference
torch.save(model.state_dict(), 'model_weights.pth')
print("✓ Model weights saved (inference only)")

# Strategy 4: Save TorchScript for production
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('production_model.pt')
print("✓ TorchScript model saved (production)")

print("\n=== Model Loading Strategies ===")

# Loading Strategy 1: Complete model (development)
loaded_complete_model = torch.load('complete_model.pth')
loaded_complete_model.eval()
print("✓ Complete model loaded")

# Loading Strategy 2: From checkpoint (training continuation)
def load_checkpoint(model, optimizer, scheduler, filepath):
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    config = checkpoint.get('model_config', {})
    
    return model, optimizer, scheduler, epoch, loss, config

# Create new model instance
new_model = ConvNet(num_classes=10)
new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.1)

# Load checkpoint
loaded_model, loaded_optimizer, loaded_scheduler, epoch, loss, config = load_checkpoint(
    new_model, new_optimizer, new_scheduler, 'model_checkpoint.pth'
)
print(f"✓ Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")

# Loading Strategy 3: Weights only for inference
inference_model = ConvNet(num_classes=10)
inference_model.load_state_dict(torch.load('model_weights.pth'))
inference_model.eval()
print("✓ Model weights loaded for inference")

# Loading Strategy 4: TorchScript for production
production_model = torch.jit.load('production_model.pt')
production_model.eval()
print("✓ TorchScript model loaded for production")
```

#### Cross-Platform Model Saving

```python
# Save model with cross-platform compatibility
def save_model_cross_platform(model, filepath, metadata=None):
    """Save model with maximum compatibility across platforms"""
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'pytorch_version': torch.__version__,
        'timestamp': torch.tensor(time.time()),
    }
    
    if metadata:
        save_dict.update(metadata)
    
    # Save with CPU mapping to ensure cross-device compatibility
    torch.save(save_dict, filepath, _use_new_zipfile_serialization=True)
    
    # Also save as TorchScript for C++ compatibility
    try:
        model.eval()
        scripted = torch.jit.script(model)
        scripted_path = filepath.replace('.pth', '_scripted.pt')
        scripted.save(scripted_path)
        print(f"✓ Cross-platform models saved: {filepath}, {scripted_path}")
    except Exception as e:
        print(f"⚠ TorchScript conversion failed: {e}")

def load_model_cross_platform(model_class, filepath, device='cpu'):
    """Load model with cross-platform compatibility"""
    
    # Load to CPU first, then move to target device
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Extract model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model and load weights
    model = model_class()
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Return additional metadata if available
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    
    return model, metadata

# Example usage
save_model_cross_platform(
    model, 
    'cross_platform_model.pth',
    metadata={
        'accuracy': 0.95,
        'dataset': 'CIFAR-10',
        'training_hours': 2.5
    }
)

# Load on different device
loaded_model, metadata = load_model_cross_platform(
    ConvNet, 
    'cross_platform_model.pth', 
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"✓ Model loaded with metadata: {metadata}")
```

### ONNX Export

**ONNX (Open Neural Network Exchange)** enables interoperability between different deep learning frameworks.

#### Exporting to ONNX

```python
import torch.onnx
import onnx
import onnxruntime

print("=== ONNX Export ===")

# Prepare model for export
model = ConvNet(num_classes=10)
model.eval()

# Create sample input
sample_input = torch.randn(1, 3, 32, 32)

# Export to ONNX
onnx_path = "convnet_model.onnx"

torch.onnx.export(
    model,                          # Model to export
    sample_input,                   # Sample input
    onnx_path,                      # Output path
    export_params=True,             # Store trained weights
    opset_version=11,               # ONNX version
    do_constant_folding=True,       # Optimize constant expressions
    input_names=['input'],          # Input names
    output_names=['output'],        # Output names
    dynamic_axes={                  # Dynamic dimensions
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✓ Model exported to ONNX: {onnx_path}")

# Verify ONNX model
try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")
    
    # Print model info
    print(f"ONNX Model IR version: {onnx_model.ir_version}")
    print(f"ONNX Model producer: {onnx_model.producer_name}")
    print(f"ONNX Model opset: {onnx_model.opset_import[0].version}")
    
except Exception as e:
    print(f"✗ ONNX model verification failed: {e}")

# Test ONNX Runtime inference
try:
    # Create ONNX Runtime session
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # Prepare input
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # Run inference
    ort_inputs = {input_name: sample_input.numpy()}
    ort_outputs = ort_session.run([output_name], ort_inputs)
    
    # Compare with PyTorch output
    with torch.no_grad():
        torch_output = model(sample_input)
    
    # Check numerical similarity
    np.testing.assert_allclose(
        torch_output.numpy(), 
        ort_outputs[0], 
        rtol=1e-03, 
        atol=1e-05
    )
    print("✓ ONNX Runtime inference matches PyTorch")
    
except Exception as e:
    print(f"✗ ONNX Runtime test failed: {e}")
```

#### Advanced ONNX Export Features

```python
# Custom ONNX export with advanced features
def export_model_to_onnx_advanced(model, sample_input, output_path):
    """Advanced ONNX export with comprehensive configuration"""
    
    # Set model to evaluation mode
    model.eval()
    
    # Export with advanced options
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        },
        # Advanced options
        keep_initializers_as_inputs=False,
        verbose=True,
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportType.ONNX
    )
    
    return output_path

# Model with custom operations
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.norm = nn.BatchNorm2d(16)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        
        # Custom operation that might need special handling
        x = torch.clamp(x, min=0, max=6)  # ReLU6
        
        return x

# Export custom model
custom_model = CustomModel()
custom_input = torch.randn(1, 3, 224, 224)

try:
    export_model_to_onnx_advanced(
        custom_model, 
        custom_input, 
        "custom_model.onnx"
    )
    print("✓ Advanced ONNX export completed")
except Exception as e:
    print(f"✗ Advanced ONNX export failed: {e}")
```

### Serialization Formats Summary

```python
# Comprehensive comparison of serialization formats
def compare_serialization_formats():
    """Compare different PyTorch serialization formats"""
    
    model = ConvNet(num_classes=10)
    sample_input = torch.randn(1, 3, 32, 32)
    
    formats = {}
    
    # 1. PyTorch state dict
    model_state = model.state_dict()
    torch.save(model_state, 'format_state_dict.pth')
    formats['State Dict'] = {
        'file': 'format_state_dict.pth',
        'size_mb': os.path.getsize('format_state_dict.pth') / (1024**2),
        'cross_platform': True,
        'needs_code': True,
        'production_ready': False
    }
    
    # 2. Complete model
    torch.save(model, 'format_complete.pth')
    formats['Complete Model'] = {
        'file': 'format_complete.pth',
        'size_mb': os.path.getsize('format_complete.pth') / (1024**2),
        'cross_platform': False,
        'needs_code': False,
        'production_ready': False
    }
    
    # 3. TorchScript
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save('format_torchscript.pt')
    formats['TorchScript'] = {
        'file': 'format_torchscript.pt',
        'size_mb': os.path.getsize('format_torchscript.pt') / (1024**2),
        'cross_platform': True,
        'needs_code': False,
        'production_ready': True
    }
    
    # 4. ONNX
    torch.onnx.export(model, sample_input, 'format_model.onnx')
    formats['ONNX'] = {
        'file': 'format_model.onnx',
        'size_mb': os.path.getsize('format_model.onnx') / (1024**2),
        'cross_platform': True,
        'needs_code': False,
        'production_ready': True
    }
    
    # Print comparison table
    print("=== Serialization Format Comparison ===")
    print(f"{'Format':<15} {'Size (MB)':<10} {'Cross-Platform':<15} {'Needs Code':<12} {'Production':<12}")
    print("-" * 70)
    
    for name, info in formats.items():
        print(f"{name:<15} {info['size_mb']:<10.2f} {str(info['cross_platform']):<15} {str(info['needs_code']):<12} {str(info['production_ready']):<12}")
    
    return formats

# Run comparison
format_comparison = compare_serialization_formats()
```

### Best Practices for Model Serialization

```python
# Production-ready model serialization best practices
class ModelManager:
    """Production model management with best practices"""
    
    @staticmethod
    def save_for_production(model, metadata, output_dir):
        """Save model with all necessary formats for production"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save state dict with metadata
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata,
            'pytorch_version': torch.__version__,
            'timestamp': time.time()
        }
        
        state_dict_path = os.path.join(output_dir, 'model_state_dict.pth')
        torch.save(checkpoint, state_dict_path)
        
        # 2. Save TorchScript
        try:
            model.eval()
            scripted = torch.jit.script(model)
            torchscript_path = os.path.join(output_dir, 'model_torchscript.pt')
            scripted.save(torchscript_path)
        except Exception as e:
            print(f"Warning: TorchScript conversion failed: {e}")
            torchscript_path = None
        
        # 3. Save ONNX if possible
        try:
            sample_input = torch.randn(1, *metadata['input_shape'])
            onnx_path = os.path.join(output_dir, 'model.onnx')
            torch.onnx.export(model, sample_input, onnx_path)
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")
            onnx_path = None
        
        # 4. Save model configuration
        config = {
            'model_class': model.__class__.__name__,
            'state_dict_path': state_dict_path,
            'torchscript_path': torchscript_path,
            'onnx_path': onnx_path,
            'metadata': metadata
        }
        
        config_path = os.path.join(output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2, default=str)
        
        print(f"✓ Production model package saved to: {output_dir}")
        return config_path

# Example usage
metadata = {
    'model_name': 'ConvNet',
    'version': '1.0',
    'accuracy': 0.95,
    'input_shape': [3, 32, 32],
    'num_classes': 10,
    'dataset': 'CIFAR-10'
}

model = ConvNet(num_classes=10)
config_path = ModelManager.save_for_production(model, metadata, 'production_model_package')
```

## Learning Objectives

By the end of this section, you should be able to:

### Core Understanding
- **Explain PyTorch's dynamic computation graph** and how it differs from static frameworks
- **Master tensor operations** including creation, manipulation, and GPU acceleration
- **Understand the autograd system** and implement custom gradient computations
- **Build neural networks** using `torch.nn` modules and custom architectures
- **Optimize models** using various optimizers and learning rate schedulers

### Production Skills
- **Deploy models using LibTorch** in C++ applications for production
- **Convert models to TorchScript** for cross-language compatibility
- **Export models to ONNX** for framework interoperability
- **Implement proper model serialization** strategies for different deployment scenarios
- **Debug and profile** PyTorch applications for performance optimization

### Advanced Capabilities
- **Design custom neural network architectures** for specific problems
- **Implement efficient data loading** pipelines using DataLoader
- **Handle GPU/CPU memory management** effectively
- **Write production-ready** model serving applications
- **Integrate PyTorch models** with existing C++ codebases

### Self-Assessment Checklist

Before proceeding to advanced topics, ensure you can:

□ Create and manipulate tensors with different data types and devices  
□ Build a custom neural network class inheriting from `nn.Module`  
□ Implement forward and backward passes with autograd  
□ Train a model using different optimizers and loss functions  
□ Save and load models using state dictionaries  
□ Convert a PyTorch model to TorchScript  
□ Export a model to ONNX format  
□ Load and run a model in C++ using LibTorch  
□ Debug gradient flow issues in complex models  
□ Profile model performance and identify bottlenecks  

### Practical Exercises

#### Exercise 1: Tensor Operations Mastery
```python
# TODO: Complete these tensor operations
import torch

# 1. Create a 3D tensor and perform reshaping operations
def tensor_reshaping_exercise():
    # Create tensor of shape (2, 3, 4)
    # Reshape to (6, 4), then to (1, 24)
    # Split back to original shape
    pass

# 2. Implement matrix operations without using built-in functions
def manual_matrix_operations(a, b):
    # Implement matrix multiplication manually using loops and tensor indexing
    # Compare with torch.mm() result
    pass

# 3. GPU memory management
def gpu_memory_exercise():
    # Create large tensors on GPU
    # Monitor memory usage
    # Implement proper cleanup
    pass
```

#### Exercise 2: Custom Neural Network Implementation
```python
# TODO: Build a complete neural network from scratch
class CustomResNet(nn.Module):
    """Implement a simplified ResNet with residual connections"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # TODO: Implement layers
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass with residual connections
        pass

# TODO: Train the network on CIFAR-10
def train_custom_resnet():
    # Implement complete training loop
    # Include data loading, loss computation, backpropagation
    # Add validation and metrics tracking
    pass
```

#### Exercise 3: Production Deployment
```python
# TODO: Create a complete model deployment pipeline
def deploy_model_pipeline():
    # 1. Train a model
    # 2. Convert to TorchScript
    # 3. Export to ONNX
    # 4. Create C++ inference application
    # 5. Benchmark performance across formats
    pass
```

#### Exercise 4: Advanced Autograd
```python
# TODO: Implement custom gradient computation
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # TODO: Implement custom forward pass
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement custom backward pass
        pass

# TODO: Use custom function in a neural network
def test_custom_autograd():
    # Test gradient computation
    # Compare with numerical gradients
    pass
```

## Study Materials

### Essential Reading
- **Primary:** "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann
- **Reference:** [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- **Advanced:** "Programming PyTorch for Deep Learning" by Ian Pointer
- **C++ Guide:** [LibTorch C++ Frontend](https://pytorch.org/cppdocs/)

### Hands-on Tutorials
- [PyTorch 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [PyTorch Recipes](https://pytorch.org/tutorials/recipes/recipes_index.html)
- [TorchScript Tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [LibTorch C++ Tutorial](https://pytorch.org/tutorials/advanced/cpp_frontend.html)

### Video Resources
- "PyTorch Fundamentals" - PyTorch Developer Day presentations
- "Deep Learning with PyTorch" - Fast.ai course materials
- "Production PyTorch" - MLOps-focused tutorials

### Practice Datasets
- **Computer Vision:** CIFAR-10, MNIST, ImageNet
- **Natural Language:** IMDB Reviews, Penn Treebank
- **Time Series:** Stock prices, sensor data
- **Custom:** Create synthetic datasets for specific architectures

### Development Environment Setup

#### Python Environment
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional packages for development
pip install tensorboard matplotlib seaborn
pip install onnx onnxruntime
pip install pytorch-lightning  # Optional: simplified training
pip install torchinfo  # Model summaries
```

#### C++ Development (LibTorch)
```bash
# Download LibTorch
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# Set environment variables
export CMAKE_PREFIX_PATH=/path/to/libtorch:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

#### Development Tools
```bash
# Install development tools
pip install black isort flake8  # Code formatting and linting
pip install pytest pytest-cov   # Testing
pip install jupyterlab         # Interactive development
pip install wandb mlflow       # Experiment tracking
```

### Performance Optimization Resources
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [TensorRT Integration](https://developer.nvidia.com/tensorrt)

### Community and Support
- **Forum:** [PyTorch Discuss](https://discuss.pytorch.org/)
- **GitHub:** [PyTorch Repository](https://github.com/pytorch/pytorch)
- **Stack Overflow:** Tag `pytorch` for specific questions
- **Discord:** PyTorch community server for real-time help

### Assessment Criteria

Your understanding will be evaluated based on:

1. **Conceptual Knowledge (30%)**
   - Understanding of dynamic vs static computation graphs
   - Knowledge of autograd mechanics
   - Awareness of performance implications

2. **Implementation Skills (40%)**
   - Ability to build custom neural networks
   - Proper use of PyTorch APIs
   - Code quality and best practices

3. **Production Readiness (20%)**
   - Model serialization and deployment
   - C++ integration capabilities
   - Performance optimization awareness

4. **Problem Solving (10%)**
   - Debugging skills
   - Architecture design decisions
   - Framework comparison insights

### Next Steps

After mastering PyTorch basics, proceed to:
- **Advanced PyTorch Techniques** (Custom operators, distributed training)
- **Model Optimization** (Quantization, pruning, knowledge distillation)
- **Production MLOps** (Model serving, monitoring, CI/CD)
- **Specialized Domains** (Computer vision, NLP, reinforcement learning)
