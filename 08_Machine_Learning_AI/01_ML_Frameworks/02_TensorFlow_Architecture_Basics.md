# TensorFlow Architecture and Basics

*Duration: 2 weeks*

## TensorFlow Overview

### Evolution (TF 1.x vs. TF 2.x)

TensorFlow has undergone significant changes between versions. Understanding these differences is crucial for modern development.

#### TensorFlow 1.x Characteristics
- **Graph-based execution**: Define-then-run paradigm
- **Session-based**: Explicit session management required
- **Complex API**: Multiple high-level APIs (tf.layers, tf.contrib, tf.slim)
- **Placeholder-heavy**: Input data through placeholders

```python
# TensorFlow 1.x Example (Legacy - Don't use in new projects)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Define computation graph
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.random.normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Create session and run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(y, feed_dict={x: input_data})
```

#### TensorFlow 2.x Improvements
- **Eager execution by default**: Immediate operation evaluation
- **Simplified API**: tf.keras as primary high-level API
- **Pythonic**: More intuitive and easier to debug
- **Function decorators**: @tf.function for graph optimization

```python
# TensorFlow 2.x Example (Modern approach)
import tensorflow as tf

# Define model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and use immediately
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train with fit() - much simpler!
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

#### Key Differences Comparison

| Aspect | TensorFlow 1.x | TensorFlow 2.x |
|--------|----------------|----------------|
| **Execution** | Graph-based (define-then-run) | Eager execution (run immediately) |
| **Sessions** | Required `tf.Session()` | Not needed |
| **API** | Multiple APIs, complex | Unified tf.keras API |
| **Debugging** | Difficult (graph inspection) | Easy (Python debugging) |
| **Learning Curve** | Steep | Gentle |
| **Performance** | Manual optimization needed | Automatic with @tf.function |

### Computation Graph Concept

The computation graph is the foundation of TensorFlow's architecture.

#### What is a Computation Graph?

A computation graph is a directed acyclic graph (DAG) where:
- **Nodes** represent operations (ops)
- **Edges** represent tensors (data flow)

```python
import tensorflow as tf
import numpy as np

# Example: Simple computation graph
@tf.function  # Creates a graph behind the scenes
def compute_function(x, y):
    # Node 1: Add operation
    add_result = tf.add(x, y)
    
    # Node 2: Multiply operation
    mul_result = tf.multiply(add_result, 2.0)
    
    # Node 3: Square operation
    square_result = tf.square(mul_result)
    
    return square_result

# Visualize the graph structure
x = tf.constant(3.0)
y = tf.constant(4.0)
result = compute_function(x, y)
print(f"Result: {result}")  # (3 + 4) * 2 = 14, 14^2 = 196

# Get concrete function to inspect graph
concrete_func = compute_function.get_concrete_function(
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32)
)
print(f"Graph operations: {[op.name for op in concrete_func.graph.get_operations()]}")
```

#### Graph Optimization Benefits

```python
# Example showing graph optimization
import time

# Regular Python function (no graph optimization)
def python_function(x):
    result = x
    for i in range(1000):
        result = result + 1
        result = result * 1.001
    return result

# TensorFlow function (graph optimized)
@tf.function
def tf_function(x):
    result = x
    for i in range(1000):
        result = result + 1
        result = result * 1.001
    return result

# Performance comparison
x = tf.constant(1.0)

# Time Python function
start = time.time()
for _ in range(100):
    python_result = python_function(x.numpy())
python_time = time.time() - start

# Time TensorFlow function
start = time.time()
for _ in range(100):
    tf_result = tf_function(x)
tf_time = time.time() - start

print(f"Python function time: {python_time:.4f}s")
print(f"TensorFlow function time: {tf_time:.4f}s")
print(f"Speedup: {python_time/tf_time:.2f}x")
```

### Eager Execution

Eager execution enables immediate evaluation of operations without building graphs.

#### Benefits of Eager Execution

```python
import tensorflow as tf

# Enable eager execution (default in TF 2.x)
print(f"Eager execution enabled: {tf.executing_eagerly()}")

# Immediate evaluation
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = a + b

print(f"Immediate result: {c}")  # No session needed!
print(f"Result as numpy: {c.numpy()}")

# Easy debugging with Python tools
import pdb

def debug_function(x):
    y = x * 2
    # pdb.set_trace()  # Standard Python debugging works!
    z = y + 1
    return z

result = debug_function(tf.constant(5.0))
print(f"Debug result: {result}")
```

#### Eager vs Graph Execution

```python
# Demonstrate both modes
import tensorflow as tf

# Eager execution example
def eager_multiply(x, y):
    print(f"Eager: Computing {x} * {y}")
    return x * y

# Graph execution example
@tf.function
def graph_multiply(x, y):
    print(f"Graph: Computing {x} * {y}")  # Only prints during tracing!
    return x * y

# Test both
print("=== Eager Execution ===")
result1 = eager_multiply(tf.constant(3.0), tf.constant(4.0))
result2 = eager_multiply(tf.constant(5.0), tf.constant(6.0))

print("\n=== Graph Execution ===")
result3 = graph_multiply(tf.constant(3.0), tf.constant(4.0))
result4 = graph_multiply(tf.constant(5.0), tf.constant(6.0))  # No print - reuses graph!

print(f"\nResults: {result1}, {result2}, {result3}, {result4}")
```

### TensorFlow Ecosystem

TensorFlow is part of a larger ecosystem of tools and libraries.

#### Core TensorFlow Components

```python
# TensorFlow ecosystem overview
import tensorflow as tf

# 1. Core TensorFlow
print(f"TensorFlow version: {tf.__version__}")

# 2. Keras (high-level API)
from tensorflow import keras
print(f"Keras version: {keras.__version__}")

# 3. TensorFlow Datasets
import tensorflow_datasets as tfds
print("Available datasets:", len(tfds.list_builders()))

# 4. TensorFlow Hub (pre-trained models)
import tensorflow_hub as hub
# Example: Load a pre-trained model
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# 5. TensorFlow Serving (model deployment)
# Used for serving models in production

# 6. TensorFlow Lite (mobile/edge deployment)
# Used for mobile and embedded devices

# 7. TensorFlow.js (browser/Node.js)
# Used for web applications
```

#### Extended Ecosystem Tools

```python
# Additional tools in the TensorFlow ecosystem

# TensorBoard for visualization
import datetime

# Set up TensorBoard logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Model with TensorBoard logging
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# To view in TensorBoard: tensorboard --logdir logs/fit

# TensorFlow Probability for probabilistic programming
try:
    import tensorflow_probability as tfp
    
    # Example: Define a normal distribution
    tfd = tfp.distributions
    normal = tfd.Normal(loc=0.0, scale=1.0)
    samples = normal.sample(1000)
    print(f"Probability samples shape: {samples.shape}")
except ImportError:
    print("TensorFlow Probability not installed")

# TensorFlow Transform for data preprocessing
try:
    import tensorflow_transform as tft
    print("TensorFlow Transform available")
except ImportError:
    print("TensorFlow Transform not installed")
```

#### Ecosystem Architecture Diagram

```
TensorFlow Ecosystem
┌─────────────────────────────────────────────────────────┐
│                     Applications                        │
├─────────────────────────────────────────────────────────┤
│ TF Serving │ TF Lite │ TF.js │ TF Extended │ TF Cloud   │
├─────────────────────────────────────────────────────────┤
│          High-Level APIs (Keras, Estimators)            │
├─────────────────────────────────────────────────────────┤
│  TF Hub  │ TF Datasets │ TF Probability │ TF Transform │
├─────────────────────────────────────────────────────────┤
│                 Core TensorFlow                         │
│            (Python/C++ APIs, Graph Engine)             │
├─────────────────────────────────────────────────────────┤
│           Hardware Abstraction Layer                   │
│         (CPU, GPU, TPU, Mobile, Edge)                  │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### Tensors and Operations

#### Understanding Tensors

Tensors are the fundamental data structure in TensorFlow - multi-dimensional arrays with a uniform type.

```python
import tensorflow as tf
import numpy as np

# Scalar (0-D tensor)
scalar = tf.constant(42)
print(f"Scalar: {scalar}, shape: {scalar.shape}, rank: {tf.rank(scalar)}")

# Vector (1-D tensor)
vector = tf.constant([1, 2, 3, 4])
print(f"Vector: {vector}, shape: {vector.shape}, rank: {tf.rank(vector)}")

# Matrix (2-D tensor)
matrix = tf.constant([[1, 2], [3, 4], [5, 6]])
print(f"Matrix: {matrix}, shape: {matrix.shape}, rank: {tf.rank(matrix)}")

# 3-D tensor (e.g., RGB image)
tensor_3d = tf.constant([[[1, 2, 3], [4, 5, 6]], 
                        [[7, 8, 9], [10, 11, 12]]])
print(f"3D Tensor: {tensor_3d.shape}, rank: {tf.rank(tensor_3d)}")

# 4-D tensor (e.g., batch of images)
tensor_4d = tf.random.normal([32, 224, 224, 3])  # batch_size, height, width, channels
print(f"4D Tensor (batch of images): {tensor_4d.shape}")
```

#### Tensor Properties and Manipulation

```python
# Tensor data types
print("=== Data Types ===")
float_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
bool_tensor = tf.constant([True, False, True], dtype=tf.bool)
string_tensor = tf.constant(["hello", "world"], dtype=tf.string)

print(f"Float tensor: {float_tensor.dtype}")
print(f"Int tensor: {int_tensor.dtype}")
print(f"Bool tensor: {bool_tensor.dtype}")
print(f"String tensor: {string_tensor.dtype}")

# Tensor operations
print("\n=== Basic Operations ===")
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# Element-wise operations
add_result = tf.add(a, b)  # or a + b
mul_result = tf.multiply(a, b)  # or a * b
print(f"Addition:\n{add_result}")
print(f"Element-wise multiplication:\n{mul_result}")

# Matrix operations
matmul_result = tf.matmul(a, b)  # or a @ b
print(f"Matrix multiplication:\n{matmul_result}")

# Reduction operations
sum_all = tf.reduce_sum(a)
sum_axis0 = tf.reduce_sum(a, axis=0)
sum_axis1 = tf.reduce_sum(a, axis=1)
print(f"Sum all: {sum_all}")
print(f"Sum axis 0: {sum_axis0}")
print(f"Sum axis 1: {sum_axis1}")
```

#### Advanced Tensor Operations

```python
# Tensor reshaping and manipulation
print("=== Tensor Reshaping ===")
original = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Original shape: {original.shape}")

# Reshape
reshaped = tf.reshape(original, [4, 2])
print(f"Reshaped to [4, 2]:\n{reshaped}")

# Transpose
transposed = tf.transpose(original)
print(f"Transposed:\n{transposed}")

# Expand dimensions
expanded = tf.expand_dims(original, axis=0)
print(f"Expanded dimensions: {expanded.shape}")

# Squeeze dimensions
squeezed = tf.squeeze(expanded, axis=0)
print(f"Squeezed back: {squeezed.shape}")

# Concatenation and stacking
print("\n=== Concatenation ===")
tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6], [7, 8]])

concat_axis0 = tf.concat([tensor1, tensor2], axis=0)
concat_axis1 = tf.concat([tensor1, tensor2], axis=1)
stacked = tf.stack([tensor1, tensor2], axis=0)

print(f"Concat axis 0:\n{concat_axis0}")
print(f"Concat axis 1:\n{concat_axis1}")
print(f"Stacked shape: {stacked.shape}")
```

### Variables and Constants

#### Constants vs Variables

```python
# Constants - immutable
print("=== Constants ===")
const = tf.constant([1, 2, 3])
print(f"Constant: {const}")
# const.assign([4, 5, 6])  # This would error - constants are immutable

# Variables - mutable, trainable parameters
print("\n=== Variables ===")
var = tf.Variable([1.0, 2.0, 3.0], name="my_variable")
print(f"Initial variable: {var}")
print(f"Variable name: {var.name}")
print(f"Variable dtype: {var.dtype}")
print(f"Variable shape: {var.shape}")

# Modifying variables
var.assign([4.0, 5.0, 6.0])
print(f"After assign: {var}")

var.assign_add([1.0, 1.0, 1.0])
print(f"After assign_add: {var}")

var.assign_sub([0.5, 0.5, 0.5])
print(f"After assign_sub: {var}")

# Variables for neural network weights
print("\n=== Neural Network Variables ===")
# Weight matrix for a dense layer: input_size=784, output_size=128
weights = tf.Variable(
    tf.random.normal([784, 128], mean=0.0, stddev=0.1),
    name="dense_weights"
)

# Bias vector
biases = tf.Variable(
    tf.zeros([128]),
    name="dense_biases"
)

print(f"Weights shape: {weights.shape}")
print(f"Biases shape: {biases.shape}")

# Trainable flag
non_trainable_var = tf.Variable([1.0, 2.0], trainable=False)
print(f"Trainable: {weights.trainable}")
print(f"Non-trainable: {non_trainable_var.trainable}")
```

#### Variable Lifecycle in Training

```python
# Example: Linear regression with variables
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convert to tensors
X_tensor = tf.constant(X)
y_tensor = tf.constant(y)

# Initialize variables (parameters to learn)
W = tf.Variable(tf.random.normal([1, 1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

print(f"Initial W: {W.numpy()}, b: {b.numpy()}")

# Define model
def linear_model(x):
    return tf.matmul(x, W) + b

# Define loss function
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Training loop
optimizer = tf.optimizers.SGD(learning_rate=0.1)

for epoch in range(200):
    with tf.GradientTape() as tape:
        y_pred = linear_model(X_tensor)
        loss = mse_loss(y_tensor, y_pred)
    
    # Compute gradients
    gradients = tape.gradient(loss, [W, b])
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, W: {W.numpy()[0,0]:.4f}, b: {b.numpy()[0]:.4f}")

print(f"Final W: {W.numpy()}, b: {b.numpy()}")
print(f"True values: W=2.0, b=1.0")
```

### Automatic Differentiation

Automatic differentiation is crucial for training neural networks via backpropagation.

#### Understanding GradientTape

```python
# Basic gradient computation
print("=== Basic Gradients ===")
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1  # y = x² + 2x + 1

# Compute gradient dy/dx = 2x + 2
gradient = tape.gradient(y, x)
print(f"x = {x.numpy()}, y = {y.numpy()}")
print(f"dy/dx = {gradient.numpy()} (expected: {2*x.numpy() + 2})")

# Multiple variables
print("\n=== Multiple Variables ===")
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = x**2 + y**2 + x*y  # z = x² + y² + xy

# Compute gradients
gradients = tape.gradient(z, [x, y])
print(f"x = {x.numpy()}, y = {y.numpy()}, z = {z.numpy()}")
print(f"dz/dx = {gradients[0].numpy()} (expected: {2*x.numpy() + y.numpy()})")
print(f"dz/dy = {gradients[1].numpy()} (expected: {2*y.numpy() + x.numpy()})")
```

#### Advanced Gradient Techniques

```python
# Persistent tape for multiple gradient computations
print("=== Persistent Tape ===")
x = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
    y = x**3
    z = y**2  # z = (x³)² = x⁶

# Can compute multiple gradients
dy_dx = tape.gradient(y, x)  # dy/dx = 3x²
dz_dx = tape.gradient(z, x)  # dz/dx = 6x⁵
dz_dy = tape.gradient(z, y)  # dz/dy = 2y

print(f"dy/dx = {dy_dx.numpy()} (expected: {3 * x.numpy()**2})")
print(f"dz/dx = {dz_dx.numpy()} (expected: {6 * x.numpy()**5})")
print(f"dz/dy = {dz_dy.numpy()} (expected: {2 * y.numpy()})")

# Clean up persistent tape
del tape

# Higher-order gradients
print("\n=== Second-Order Gradients ===")
x = tf.Variable(2.0)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = x**4
    
    # First derivative
    dy_dx = inner_tape.gradient(y, x)

# Second derivative
d2y_dx2 = outer_tape.gradient(dy_dx, x)

print(f"y = x⁴, dy/dx = {dy_dx.numpy()} (expected: {4 * x.numpy()**3})")
print(f"d²y/dx² = {d2y_dx2.numpy()} (expected: {12 * x.numpy()**2})")
```

#### Practical Neural Network Gradients

```python
# Neural network gradient example
print("=== Neural Network Gradients ===")

# Simple neural network
class SimpleNN(tf.Module):
    def __init__(self):
        self.w1 = tf.Variable(tf.random.normal([2, 3]))
        self.b1 = tf.Variable(tf.zeros([3]))
        self.w2 = tf.Variable(tf.random.normal([3, 1]))
        self.b2 = tf.Variable(tf.zeros([1]))
    
    def __call__(self, x):
        hidden = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        output = tf.matmul(hidden, self.w2) + self.b2
        return output

# Create model and data
model = SimpleNN()
x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
y_true = tf.constant([[1.5]], dtype=tf.float32)

# Forward pass with gradient tracking
with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = tf.reduce_mean(tf.square(y_true - y_pred))

# Compute gradients
variables = [model.w1, model.b1, model.w2, model.b2]
gradients = tape.gradient(loss, variables)

print(f"Loss: {loss.numpy()}")
for i, (var, grad) in enumerate(zip(variables, gradients)):
    print(f"Variable {i} shape: {var.shape}, Gradient shape: {grad.shape}")
    print(f"Gradient norm: {tf.norm(grad).numpy():.6f}")
```

### Metrics and Losses

#### Common Loss Functions

```python
import tensorflow as tf
import numpy as np

# Generate sample data
y_true_classification = tf.constant([0, 1, 2, 1, 0])  # Class labels
y_pred_classification = tf.constant([[0.8, 0.1, 0.1],  # Logits
                                   [0.2, 0.7, 0.1],
                                   [0.1, 0.2, 0.7],
                                   [0.3, 0.6, 0.1],
                                   [0.9, 0.05, 0.05]])

y_true_regression = tf.constant([2.5, 1.8, 3.2, 0.9])
y_pred_regression = tf.constant([2.3, 1.9, 3.0, 1.1])

print("=== Classification Losses ===")

# Sparse categorical crossentropy
sparse_ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
    y_true_classification, y_pred_classification, from_logits=True
)
print(f"Sparse Categorical Crossentropy: {sparse_ce_loss.numpy()}")

# Binary crossentropy
y_true_binary = tf.constant([0, 1, 1, 0])
y_pred_binary = tf.constant([0.1, 0.9, 0.8, 0.2])
binary_ce_loss = tf.keras.losses.binary_crossentropy(y_true_binary, y_pred_binary)
print(f"Binary Crossentropy: {binary_ce_loss.numpy()}")

print("\n=== Regression Losses ===")

# Mean Squared Error
mse_loss = tf.keras.losses.mean_squared_error(y_true_regression, y_pred_regression)
print(f"MSE Loss: {mse_loss.numpy()}")

# Mean Absolute Error
mae_loss = tf.keras.losses.mean_absolute_error(y_true_regression, y_pred_regression)
print(f"MAE Loss: {mae_loss.numpy()}")

# Huber Loss (robust to outliers)
huber_loss = tf.keras.losses.huber(y_true_regression, y_pred_regression, delta=1.0)
print(f"Huber Loss: {huber_loss.numpy()}")
```

#### Custom Loss Functions

```python
# Custom loss function example
def custom_weighted_mse(y_true, y_pred, weights):
    """Weighted MSE loss"""
    squared_diff = tf.square(y_true - y_pred)
    weighted_squared_diff = squared_diff * weights
    return tf.reduce_mean(weighted_squared_diff)

# Example usage
weights = tf.constant([1.0, 2.0, 0.5, 1.5])  # Different weights for different samples
custom_loss = custom_weighted_mse(y_true_regression, y_pred_regression, weights)
print(f"Custom Weighted MSE: {custom_loss.numpy()}")

# Focal loss for imbalanced classification
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for addressing class imbalance"""
    # Convert to probabilities
    y_pred = tf.nn.softmax(y_pred)
    
    # One-hot encode y_true
    y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
    
    # Compute focal loss
    ce = -y_true_onehot * tf.math.log(y_pred + 1e-8)
    pt = tf.where(y_true_onehot == 1, y_pred, 1 - y_pred)
    focal_weight = alpha * tf.pow(1 - pt, gamma)
    focal_loss_val = focal_weight * ce
    
    return tf.reduce_sum(focal_loss_val, axis=1)

focal_loss_result = focal_loss(y_true_classification, y_pred_classification)
print(f"Focal Loss: {focal_loss_result.numpy()}")
```

#### Metrics for Model Evaluation

```python
print("=== Classification Metrics ===")

# Accuracy
accuracy = tf.keras.metrics.sparse_categorical_accuracy(
    y_true_classification, y_pred_classification
)
print(f"Accuracy: {accuracy.numpy()}")

# Precision and Recall (for binary classification)
y_true_binary_int = tf.cast(y_true_binary, tf.int32)
y_pred_binary_rounded = tf.cast(y_pred_binary > 0.5, tf.int32)

precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()

precision_metric.update_state(y_true_binary_int, y_pred_binary_rounded)
recall_metric.update_state(y_true_binary_int, y_pred_binary_rounded)

print(f"Precision: {precision_metric.result().numpy()}")
print(f"Recall: {recall_metric.result().numpy()}")

print("\n=== Regression Metrics ===")

# R² Score
def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

r2 = r2_score(y_true_regression, y_pred_regression)
print(f"R² Score: {r2.numpy()}")

# Root Mean Squared Error
rmse = tf.keras.metrics.RootMeanSquaredError()
rmse.update_state(y_true_regression, y_pred_regression)
print(f"RMSE: {rmse.result().numpy()}")

# Custom streaming metric
class MeanAbsolutePercentageError(tf.keras.metrics.Metric):
    def __init__(self, name='mape', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_percentage_error = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        percentage_error = tf.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
        self.total_percentage_error.assign_add(tf.reduce_sum(percentage_error))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total_percentage_error / self.count
    
    def reset_state(self):
        self.total_percentage_error.assign(0)
        self.count.assign(0)

# Use custom metric
mape = MeanAbsolutePercentageError()
mape.update_state(y_true_regression, y_pred_regression)
print(f"MAPE: {mape.result().numpy()}%")
```

## TensorFlow C++ API

### TensorFlow C++ Architecture

TensorFlow's C++ API provides high-performance inference and training capabilities, essential for production systems and embedded applications.

#### Architecture Overview

```
TensorFlow C++ API Architecture
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
├─────────────────────────────────────────────────────────┤
│  C++ Client API  │  Session Interface │  SavedModel API │
├─────────────────────────────────────────────────────────┤
│         TensorFlow Core (C++)                          │
│    Graph Definition │ Operation Registry │ Kernels     │
├─────────────────────────────────────────────────────────┤
│              Device Abstraction Layer                  │
│           CPU │ GPU (CUDA) │ TPU │ Custom              │
└─────────────────────────────────────────────────────────┘
```

#### Key Components

**1. Core Headers and Namespaces**

```cpp
// Essential TensorFlow C++ headers
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

// For SavedModel loading
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/constants.h>
#include <tensorflow/cc/saved_model/signature_constants.h>

using namespace tensorflow;
using namespace tensorflow::ops;
```

**2. Basic C++ TensorFlow Program Structure**

```cpp
#include <iostream>
#include <vector>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

int main() {
    // Create a root scope
    Scope root = Scope::NewRootScope();
    
    // Define computation graph
    auto A = Const(root, {{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto B = Const(root, {{5.0f, 6.0f}, {7.0f, 8.0f}});
    auto C = MatMul(root, A, B);
    
    // Create client session
    ClientSession session(root);
    
    // Run computation
    std::vector<Tensor> outputs;
    Status status = session.Run({C}, &outputs);
    
    if (status.ok()) {
        std::cout << "Result:\n" << outputs[0].matrix<float>() << std::endl;
    } else {
        std::cout << "Error: " << status.ToString() << std::endl;
    }
    
    return 0;
}
```

### Building and Linking

#### Building TensorFlow C++ from Source

```bash
# Install Bazel (TensorFlow's build system)
# On Ubuntu/Debian:
sudo apt install bazel

# Clone TensorFlow repository
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

# Configure build
./configure

# Build TensorFlow C++ library
bazel build //tensorflow:libtensorflow_cc.so

# Build specific targets
bazel build //tensorflow/cc:cc_ops
bazel build //tensorflow/core:tensorflow
```

#### CMake Integration Example

```cmake
# CMakeLists.txt for TensorFlow C++ project
cmake_minimum_required(VERSION 3.10)
project(TensorFlowCppExample)

set(CMAKE_CXX_STANDARD 17)

# Find TensorFlow
find_package(TensorFlow REQUIRED)

# Or manually specify paths
set(TENSORFLOW_ROOT "/path/to/tensorflow")
set(TENSORFLOW_INCLUDE_DIRS 
    "${TENSORFLOW_ROOT}"
    "${TENSORFLOW_ROOT}/bazel-genfiles"
    "${TENSORFLOW_ROOT}/tensorflow/contrib/makefile/gen/protobuf/include"
)

set(TENSORFLOW_LIBRARIES
    "${TENSORFLOW_ROOT}/bazel-bin/tensorflow/libtensorflow_cc.so"
    "${TENSORFLOW_ROOT}/bazel-bin/tensorflow/libtensorflow_framework.so"
)

# Include directories
include_directories(${TENSORFLOW_INCLUDE_DIRS})

# Create executable
add_executable(tf_example main.cpp)

# Link libraries
target_link_libraries(tf_example ${TENSORFLOW_LIBRARIES})

# Compiler flags
target_compile_options(tf_example PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
```

#### Compilation Commands

```bash
# Direct compilation with g++
g++ -std=c++17 \
    -I/path/to/tensorflow \
    -I/path/to/tensorflow/bazel-genfiles \
    -I/path/to/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include \
    -L/path/to/tensorflow/bazel-bin/tensorflow \
    -ltensorflow_cc -ltensorflow_framework \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    main.cpp -o tf_example

# Using pkg-config (if available)
g++ -std=c++17 `pkg-config --cflags --libs tensorflow-cc` main.cpp -o tf_example
```

### Session Management

#### Traditional Session API

```cpp
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/graph.pb.h>

class TensorFlowInference {
private:
    std::unique_ptr<Session> session;
    
public:
    bool LoadGraph(const std::string& graph_path) {
        // Read graph definition
        GraphDef graph_def;
        Status status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
        if (!status.ok()) {
            std::cerr << "Failed to load graph: " << status.ToString() << std::endl;
            return false;
        }
        
        // Create session
        session.reset(NewSession(SessionOptions()));
        
        // Add graph to session
        status = session->Create(graph_def);
        if (!status.ok()) {
            std::cerr << "Failed to create session: " << status.ToString() << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool RunInference(const std::vector<std::pair<std::string, Tensor>>& inputs,
                     const std::vector<std::string>& output_names,
                     std::vector<Tensor>* outputs) {
        Status status = session->Run(inputs, output_names, {}, outputs);
        if (!status.ok()) {
            std::cerr << "Inference failed: " << status.ToString() << std::endl;
            return false;
        }
        return true;
    }
    
    ~TensorFlowInference() {
        if (session) {
            session->Close();
        }
    }
};

// Usage example
int main() {
    TensorFlowInference inference;
    
    if (!inference.LoadGraph("model.pb")) {
        return -1;
    }
    
    // Prepare input
    Tensor input_tensor(DT_FLOAT, TensorShape({1, 224, 224, 3}));
    auto input_matrix = input_tensor.tensor<float, 4>();
    
    // Fill with sample data
    for (int i = 0; i < 224; ++i) {
        for (int j = 0; j < 224; ++j) {
            for (int k = 0; k < 3; ++k) {
                input_matrix(0, i, j, k) = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }
    
    // Run inference
    std::vector<Tensor> outputs;
    std::vector<std::pair<std::string, Tensor>> inputs = {{"input:0", input_tensor}};
    
    if (inference.RunInference(inputs, {"output:0"}, &outputs)) {
        std::cout << "Inference successful!" << std::endl;
        std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
    }
    
    return 0;
}
```

#### Modern ClientSession API

```cpp
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>

class ModernTensorFlowSession {
private:
    Scope root;
    std::unique_ptr<ClientSession> session;
    
public:
    ModernTensorFlowSession() : root(Scope::NewRootScope()) {
        session = std::make_unique<ClientSession>(root);
    }
    
    // Build a simple neural network
    Output BuildNetwork(const Output& input) {
        // First layer: Dense with ReLU
        auto weights1 = Variable(root.WithOpName("weights1"), 
                                {784, 128}, DT_FLOAT);
        auto biases1 = Variable(root.WithOpName("biases1"), 
                               {128}, DT_FLOAT);
        
        auto layer1 = Add(root, MatMul(root, input, weights1), biases1);
        auto relu1 = Relu(root, layer1);
        
        // Second layer: Output
        auto weights2 = Variable(root.WithOpName("weights2"), 
                                {128, 10}, DT_FLOAT);
        auto biases2 = Variable(root.WithOpName("biases2"), 
                               {10}, DT_FLOAT);
        
        auto output = Add(root, MatMul(root, relu1, weights2), biases2);
        return output;
    }
    
    bool InitializeVariables() {
        // Initialize all variables
        auto init_op = ops::InitializeOp(root, 
            {ops::Variable(root.WithOpName("weights1"), {784, 128}, DT_FLOAT),
             ops::Variable(root.WithOpName("biases1"), {128}, DT_FLOAT),
             ops::Variable(root.WithOpName("weights2"), {128, 10}, DT_FLOAT),
             ops::Variable(root.WithOpName("biases2"), {10}, DT_FLOAT)});
        
        Status status = session->Run({}, {}, {init_op}, nullptr);
        return status.ok();
    }
    
    bool Forward(const Tensor& input_data, Tensor* output) {
        auto input_placeholder = Placeholder(root, DT_FLOAT);
        auto network_output = BuildNetwork(input_placeholder);
        
        std::vector<Tensor> outputs;
        Status status = session->Run({{input_placeholder, input_data}}, 
                                    {network_output}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            *output = outputs[0];
            return true;
        }
        return false;
    }
};
```

### Graph Definition and Execution

#### Building Computation Graphs

```cpp
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/framework/scope.h>

class GraphBuilder {
private:
    Scope root;
    
public:
    GraphBuilder() : root(Scope::NewRootScope()) {}
    
    // Build a convolutional neural network graph
    Output BuildCNN(const Output& input) {
        // Input shape: [batch, height, width, channels]
        
        // Convolutional layer 1
        auto conv1_filter = Variable(root.WithOpName("conv1_filter"), 
                                   {5, 5, 3, 32}, DT_FLOAT);
        auto conv1 = Conv2D(root, input, conv1_filter, 
                           {1, 1, 1, 1}, "SAME");
        auto relu1 = Relu(root, conv1);
        auto pool1 = MaxPool(root, relu1, {1, 2, 2, 1}, 
                           {1, 2, 2, 1}, "SAME");
        
        // Convolutional layer 2
        auto conv2_filter = Variable(root.WithOpName("conv2_filter"), 
                                   {5, 5, 32, 64}, DT_FLOAT);
        auto conv2 = Conv2D(root, pool1, conv2_filter, 
                           {1, 1, 1, 1}, "SAME");
        auto relu2 = Relu(root, conv2);
        auto pool2 = MaxPool(root, relu2, {1, 2, 2, 1}, 
                           {1, 2, 2, 1}, "SAME");
        
        // Flatten for dense layer
        auto flatten = Reshape(root, pool2, {-1, 56 * 56 * 64});
        
        // Dense layer
        auto dense_weights = Variable(root.WithOpName("dense_weights"), 
                                    {56 * 56 * 64, 1024}, DT_FLOAT);
        auto dense_biases = Variable(root.WithOpName("dense_biases"), 
                                   {1024}, DT_FLOAT);
        auto dense = Add(root, MatMul(root, flatten, dense_weights), dense_biases);
        auto relu3 = Relu(root, dense);
        
        // Output layer
        auto output_weights = Variable(root.WithOpName("output_weights"), 
                                     {1024, 10}, DT_FLOAT);
        auto output_biases = Variable(root.WithOpName("output_biases"), 
                                    {10}, DT_FLOAT);
        auto output = Add(root, MatMul(root, relu3, output_weights), output_biases);
        
        return output;
    }
    
    // Build a custom operation
    Output CustomLinearTransform(const Output& input, 
                                const std::vector<int>& input_shape,
                                const std::vector<int>& output_shape) {
        int input_size = 1, output_size = 1;
        for (int dim : input_shape) input_size *= dim;
        for (int dim : output_shape) output_size *= dim;
        
        auto weights = Variable(root.WithOpName("custom_weights"), 
                              {input_size, output_size}, DT_FLOAT);
        auto biases = Variable(root.WithOpName("custom_biases"), 
                             {output_size}, DT_FLOAT);
        
        auto flattened = Reshape(root, input, {-1, input_size});
        auto linear = Add(root, MatMul(root, flattened, weights), biases);
        auto reshaped = Reshape(root, linear, output_shape);
        
        return reshaped;
    }
    
    Scope& GetScope() { return root; }
};

// Graph execution example
void ExecuteGraph() {
    GraphBuilder builder;
    
    // Create input placeholder
    auto input = Placeholder(builder.GetScope(), DT_FLOAT, 
                           Placeholder::Shape({-1, 224, 224, 3}));
    
    // Build network
    auto output = builder.BuildCNN(input);
    
    // Create session and run
    ClientSession session(builder.GetScope());
    
    // Initialize variables
    auto init_vars = ops::InitializeOp(builder.GetScope(), 
                                      ops::Variable::GetAll(builder.GetScope()));
    session.Run({}, {}, {init_vars}, nullptr);
    
    // Prepare input data
    Tensor input_tensor(DT_FLOAT, TensorShape({1, 224, 224, 3}));
    // ... fill input_tensor with data ...
    
    // Run inference
    std::vector<Tensor> outputs;
    Status status = session.Run({{input, input_tensor}}, {output}, &outputs);
    
    if (status.ok()) {
        std::cout << "Graph execution successful!" << std::endl;
        std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
    }
}
```

### Loading and Running Models

#### SavedModel Loading in C++

```cpp
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/constants.h>

class SavedModelInference {
private:
    SavedModelBundle model_bundle;
    bool model_loaded;
    
public:
    SavedModelInference() : model_loaded(false) {}
    
    bool LoadModel(const std::string& model_path, 
                   const std::vector<std::string>& tags = {kSavedModelTagServe}) {
        SessionOptions session_options;
        RunOptions run_options;
        
        Status status = LoadSavedModel(session_options, run_options,
                                      model_path, tags, &model_bundle);
        
        if (!status.ok()) {
            std::cerr << "Failed to load model: " << status.ToString() << std::endl;
            return false;
        }
        
        model_loaded = true;
        std::cout << "Model loaded successfully from: " << model_path << std::endl;
        
        // Print available signatures
        PrintSignatures();
        
        return true;
    }
    
    void PrintSignatures() {
        if (!model_loaded) return;
        
        std::cout << "Available signatures:" << std::endl;
        for (const auto& signature_pair : model_bundle.GetSignatures()) {
            std::cout << "  - " << signature_pair.first << std::endl;
            
            const SignatureDef& signature = signature_pair.second;
            
            std::cout << "    Inputs:" << std::endl;
            for (const auto& input : signature.inputs()) {
                std::cout << "      " << input.first << ": " 
                         << input.second.name() << " ("
                         << DataType_Name(input.second.dtype()) << ")" << std::endl;
            }
            
            std::cout << "    Outputs:" << std::endl;
            for (const auto& output : signature.outputs()) {
                std::cout << "      " << output.first << ": " 
                         << output.second.name() << " ("
                         << DataType_Name(output.second.dtype()) << ")" << std::endl;
            }
        }
    }
    
    bool RunInference(const std::string& signature_name,
                     const std::vector<std::pair<std::string, Tensor>>& inputs,
                     const std::vector<std::string>& output_names,
                     std::vector<Tensor>* outputs) {
        if (!model_loaded) {
            std::cerr << "Model not loaded!" << std::endl;
            return false;
        }
        
        // Get signature
        const auto& signatures = model_bundle.GetSignatures();
        auto signature_it = signatures.find(signature_name);
        if (signature_it == signatures.end()) {
            std::cerr << "Signature '" << signature_name << "' not found!" << std::endl;
            return false;
        }
        
        const SignatureDef& signature = signature_it->second;
        
        // Prepare input tensors with correct names
        std::vector<std::pair<std::string, Tensor>> feed_tensors;
        for (const auto& input_pair : inputs) {
            const std::string& input_key = input_pair.first;
            const Tensor& input_tensor = input_pair.second;
            
            auto input_it = signature.inputs().find(input_key);
            if (input_it != signature.inputs().end()) {
                feed_tensors.push_back({input_it->second.name(), input_tensor});
            } else {
                std::cerr << "Input '" << input_key << "' not found in signature!" << std::endl;
                return false;
            }
        }
        
        // Prepare output tensor names
        std::vector<std::string> fetch_tensors;
        for (const std::string& output_key : output_names) {
            auto output_it = signature.outputs().find(output_key);
            if (output_it != signature.outputs().end()) {
                fetch_tensors.push_back(output_it->second.name());
            } else {
                std::cerr << "Output '" << output_key << "' not found in signature!" << std::endl;
                return false;
            }
        }
        
        // Run inference
        Status status = model_bundle.GetSession()->Run(feed_tensors, fetch_tensors, 
                                                      {}, outputs);
        
        if (!status.ok()) {
            std::cerr << "Inference failed: " << status.ToString() << std::endl;
            return false;
        }
        
        return true;
    }
};

// Usage example
int main() {
    SavedModelInference inference;
    
    // Load model
    if (!inference.LoadModel("/path/to/saved_model")) {
        return -1;
    }
    
    // Prepare input
    Tensor input_tensor(DT_FLOAT, TensorShape({1, 224, 224, 3}));
    auto input_flat = input_tensor.flat<float>();
    
    // Fill with random data (in practice, use real image data)
    for (int i = 0; i < input_flat.size(); ++i) {
        input_flat(i) = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Run inference
    std::vector<Tensor> outputs;
    std::vector<std::pair<std::string, Tensor>> inputs = {{"input_image", input_tensor}};
    
    if (inference.RunInference("serving_default", inputs, {"predictions"}, &outputs)) {
        std::cout << "Inference successful!" << std::endl;
        std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;
        
        // Print predictions
        auto predictions = outputs[0].flat<float>();
        std::cout << "Predictions: ";
        for (int i = 0; i < std::min(10, static_cast<int>(predictions.size())); ++i) {
            std::cout << predictions(i) << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
```

## TensorFlow Model Representation

### SavedModel Format

SavedModel is TensorFlow's universal serialization format, designed for production deployment across different platforms and languages.

#### SavedModel Structure

```
saved_model/
├── saved_model.pb          # MetaGraphDef protocol buffer
├── variables/              # Variable checkpoints
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/                 # Additional files (vocabulary, etc.)
    └── vocabulary.txt
```

#### Creating and Saving Models

```python
import tensorflow as tf
import numpy as np
import os

# Create a simple model
class SimpleModel(tf.Module):
    def __init__(self):
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
    def predict(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 10], dtype=tf.float32)
    ])
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.predict(x)
            loss = tf.keras.losses.categorical_crossentropy(y, predictions)
            loss = tf.reduce_mean(loss)
        
        # Get trainable variables
        trainable_vars = [self.dense1.kernel, self.dense1.bias,
                         self.dense2.kernel, self.dense2.bias,
                         self.output_layer.kernel, self.output_layer.bias]
        
        gradients = tape.gradient(loss, trainable_vars)
        
        # Apply gradients (simplified optimizer)
        learning_rate = 0.001
        for var, grad in zip(trainable_vars, gradients):
            if grad is not None:
                var.assign_sub(learning_rate * grad)
        
        return loss

# Create and train model
model = SimpleModel()

# Generate dummy training data
x_train = tf.random.normal([1000, 784])
y_train = tf.random.uniform([1000, 10], maxval=2, dtype=tf.int32)
y_train = tf.cast(y_train, tf.float32)

print("Training model...")
for epoch in range(10):
    loss = model.train_step(x_train, y_train)
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Save the model
save_path = "./my_saved_model"
tf.saved_model.save(model, save_path)
print(f"Model saved to {save_path}")

# Inspect saved signatures
loaded_model = tf.saved_model.load(save_path)
print("Available signatures:")
for signature_name, signature in loaded_model.signatures.items():
    print(f"  {signature_name}: {signature}")
```

#### Loading and Using SavedModels

```python
# Load SavedModel
print("=== Loading SavedModel ===")
loaded_model = tf.saved_model.load(save_path)

# Get the prediction function
predict_fn = loaded_model.signatures['serving_default']

# Inspect the signature
print(f"Input signature: {predict_fn.structured_input_signature}")
print(f"Output signature: {predict_fn.structured_outputs}")

# Make predictions
test_input = tf.random.normal([5, 784])
predictions = predict_fn(x=test_input)

print(f"Input shape: {test_input.shape}")
print(f"Predictions shape: {predictions['output_0'].shape}")
print(f"Sample predictions:\n{predictions['output_0'][:3]}")

# Alternative: Using tf.keras.models.load_model for Keras models
# For Keras models, you can also use:
# keras_model = tf.keras.models.load_model(save_path)
```

#### Advanced SavedModel Features

```python
# Model with multiple signatures and preprocessing
class AdvancedModel(tf.Module):
    def __init__(self):
        self.preprocessing_layer = tf.keras.layers.Normalization()
        self.model_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Fit normalization layer with dummy data
        dummy_data = tf.random.normal([100, 784])
        self.preprocessing_layer.adapt(dummy_data)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
    def serve(self, inputs):
        """Main serving signature"""
        normalized = self.preprocessing_layer(inputs)
        return self.model_layers(normalized)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
    def serve_raw(self, inputs):
        """Serving without preprocessing"""
        return self.model_layers(inputs)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32)])
    def serve_images(self, images):
        """Serving signature for image inputs"""
        # Flatten images
        flattened = tf.reshape(images, [-1, 784])
        return self.serve(flattened)

# Create and save advanced model
advanced_model = AdvancedModel()

# Dummy training
x_dummy = tf.random.normal([100, 784])
y_dummy = advanced_model.serve(x_dummy)  # Just to initialize layers

# Save with multiple signatures
advanced_save_path = "./advanced_saved_model"
tf.saved_model.save(
    advanced_model,
    advanced_save_path,
    signatures={
        'serving_default': advanced_model.serve,
        'serve_raw': advanced_model.serve_raw,
        'serve_images': advanced_model.serve_images
    }
)

print(f"Advanced model saved to {advanced_save_path}")

# Load and inspect
advanced_loaded = tf.saved_model.load(advanced_save_path)
print("Available signatures:")
for name, sig in advanced_loaded.signatures.items():
    print(f"  {name}")
    print(f"    Input: {sig.structured_input_signature[1]}")
    print(f"    Output: {sig.structured_outputs}")
```

### Protocol Buffers

Protocol Buffers (protobuf) are used extensively in TensorFlow for serializing structured data.

#### Understanding TensorFlow Protobufs

```python
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2

# Example: Working with GraphDef protobuf
def analyze_graph_protobuf(saved_model_path):
    """Analyze the graph structure from SavedModel"""
    
    # Load SavedModel
    loaded_model = tf.saved_model.load(saved_model_path)
    
    # Get the concrete function
    concrete_func = loaded_model.signatures['serving_default']
    graph_def = concrete_func.graph.as_graph_def()
    
    print(f"Graph contains {len(graph_def.node)} nodes")
    
    # Analyze node types
    node_types = {}
    for node in graph_def.node:
        op_type = node.op
        node_types[op_type] = node_types.get(op_type, 0) + 1
    
    print("Node types:")
    for op_type, count in sorted(node_types.items()):
        print(f"  {op_type}: {count}")
    
    # Print first few nodes
    print("\nFirst few nodes:")
    for i, node in enumerate(graph_def.node[:5]):
        print(f"  {i}: {node.name} ({node.op})")
        if node.input:
            print(f"     Inputs: {list(node.input)}")
    
    return graph_def

# Analyze the previously saved model
if os.path.exists(save_path):
    graph_def = analyze_graph_protobuf(save_path)
```

#### Custom Protobuf Manipulation

```python
# Example: Modifying graph protobuf
def modify_graph_protobuf(graph_def):
    """Example of graph modification"""
    from tensorflow.python.framework import dtypes
    from tensorflow.core.framework import attr_value_pb2
    
    # Create a new constant node
    new_node = node_def_pb2.NodeDef()
    new_node.name = "custom_constant"
    new_node.op = "Const"
    
    # Set the value attribute
    dtype_attr = attr_value_pb2.AttrValue()
    dtype_attr.type = dtypes.float32.as_datatype_enum
    new_node.attr["dtype"].CopyFrom(dtype_attr)
    
    # Add tensor value
    tensor_attr = attr_value_pb2.AttrValue()
    tensor_attr.tensor.dtype = dtypes.float32.as_datatype_enum
    tensor_attr.tensor.tensor_shape.dim.add().size = 1
    tensor_attr.tensor.float_val.append(42.0)
    new_node.attr["value"].CopyFrom(tensor_attr)
    
    # Add to graph
    modified_graph = graph_pb2.GraphDef()
    modified_graph.CopyFrom(graph_def)
    modified_graph.node.append(new_node)
    
    print(f"Added node '{new_node.name}' to graph")
    return modified_graph

# Working with MetaGraph
def create_meta_graph_example():
    """Create a MetaGraph protobuf example"""
    
    # Create a simple graph
    with tf.Graph().as_default() as graph:
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name="input")
        w = tf.Variable(tf.random.normal([784, 10]), name="weights")
        b = tf.Variable(tf.zeros([10]), name="biases")
        y = tf.add(tf.matmul(x, w), b, name="output")
        
        # Create session
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # Create MetaGraphDef
            meta_graph_def = tf.compat.v1.train.export_meta_graph()
            
            print(f"MetaGraph has {len(meta_graph_def.graph_def.node)} nodes")
            print(f"Collection keys: {list(meta_graph_def.collection_def.keys())}")
            
            # Analyze signature def (if present)
            if meta_graph_def.signature_def:
                for sig_name, sig_def in meta_graph_def.signature_def.items():
                    print(f"Signature '{sig_name}':")
                    print(f"  Inputs: {list(sig_def.inputs.keys())}")
                    print(f"  Outputs: {list(sig_def.outputs.keys())}")
            
            return meta_graph_def

# Create example MetaGraph
meta_graph = create_meta_graph_example()
```

### GraphDef and MetaGraphDef

#### Understanding GraphDef

```python
# GraphDef analysis and manipulation
def analyze_graphdef_structure():
    """Detailed analysis of GraphDef structure"""
    
    # Create a computation graph
    @tf.function
    def complex_computation(x):
        # Multiple operations to create interesting graph
        conv = tf.nn.conv2d(x, tf.random.normal([3, 3, 1, 32]), 
                           strides=1, padding='SAME')
        relu = tf.nn.relu(conv)
        pool = tf.nn.max_pool2d(relu, ksize=2, strides=2, padding='SAME')
        flat = tf.reshape(pool, [-1, tf.reduce_prod(tf.shape(pool)[1:])])
        dense = tf.matmul(flat, tf.random.normal([tf.shape(flat)[1], 10]))
        return tf.nn.softmax(dense)
    
    # Get concrete function and graph
    concrete_func = complex_computation.get_concrete_function(
        tf.TensorSpec([None, 28, 28, 1], tf.float32)
    )
    
    graph_def = concrete_func.graph.as_graph_def()
    
    print("=== GraphDef Analysis ===")
    print(f"Total nodes: {len(graph_def.node)}")
    
    # Categorize nodes
    categories = {
        'Variables': [],
        'Constants': [],
        'Operations': [],
        'Placeholders': []
    }
    
    for node in graph_def.node:
        if node.op == 'VarHandleOp':
            categories['Variables'].append(node.name)
        elif node.op == 'Const':
            categories['Constants'].append(node.name)
        elif node.op == 'Placeholder':
            categories['Placeholders'].append(node.name)
        else:
            categories['Operations'].append(node.name)
    
    for category, nodes in categories.items():
        print(f"{category}: {len(nodes)}")
        if nodes:
            print(f"  Examples: {nodes[:3]}")
    
    # Analyze dependencies
    print("\n=== Node Dependencies ===")
    for node in graph_def.node[:5]:  # First 5 nodes
        print(f"{node.name} ({node.op}):")
        if node.input:
            print(f"  Depends on: {list(node.input)}")
        else:
            print("  No dependencies (source node)")
    
    return graph_def

graph_def_example = analyze_graphdef_structure()
```

#### MetaGraphDef Structure

```python
# Working with MetaGraphDef
def create_and_analyze_metagraph():
    """Create and analyze MetaGraphDef"""
    
    # Build a model with Keras
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Dummy training
    x_train = np.random.random((100, 784))
    y_train = np.random.randint(0, 10, (100,))
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    # Save and load to get MetaGraphDef
    temp_path = "./temp_model"
    model.save(temp_path)
    
    # Load SavedModel and extract MetaGraphDef
    saved_model = tf.saved_model.load(temp_path)
    
    # For more detailed MetaGraph analysis, we need to use the lower-level API
    from tensorflow.python.saved_model import loader
    from tensorflow.python.saved_model import tag_constants
    
    print("=== MetaGraphDef Analysis ===")
    
    # Load using the loader to get MetaGraphDef
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            meta_graph_def = loader.load(sess, [tag_constants.SERVING], temp_path)
            
            print(f"GraphDef nodes: {len(meta_graph_def.graph_def.node)}")
            print(f"Signature definitions: {len(meta_graph_def.signature_def)}")
            
            # Analyze signatures
            for sig_name, sig_def in meta_graph_def.signature_def.items():
                print(f"\nSignature: {sig_name}")
                print(f"  Method: {sig_def.method_name}")
                
                print("  Inputs:")
                for input_name, input_info in sig_def.inputs.items():
                    print(f"    {input_name}: {input_info.name} "
                         f"({tf.dtypes.as_dtype(input_info.dtype).name})")
                
                print("  Outputs:")
                for output_name, output_info in sig_def.outputs.items():
                    print(f"    {output_name}: {output_info.name} "
                         f"({tf.dtypes.as_dtype(output_info.dtype).name})")
            
            # Analyze collections
            print(f"\nCollections: {len(meta_graph_def.collection_def)}")
            for collection_name, collection_def in meta_graph_def.collection_def.items():
                print(f"  {collection_name}: {len(collection_def.bytes_list.value)} items")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_path)
    
    return meta_graph_def

meta_graph_example = create_and_analyze_metagraph()
```

### Checkpoint Files

#### Understanding TensorFlow Checkpoints

```python
# Working with checkpoints
def checkpoint_example():
    """Demonstrate checkpoint creation and loading"""
    
    # Create a simple model
    class CheckpointModel(tf.Module):
        def __init__(self):
            self.dense1 = tf.Variable(tf.random.normal([784, 128]), name="dense1_weights")
            self.bias1 = tf.Variable(tf.zeros([128]), name="dense1_bias")
            self.dense2 = tf.Variable(tf.random.normal([128, 10]), name="dense2_weights")
            self.bias2 = tf.Variable(tf.zeros([10]), name="dense2_bias")
            self.global_step = tf.Variable(0, dtype=tf.int64, name="global_step")
        
        @tf.function
        def forward(self, x):
            hidden = tf.nn.relu(tf.matmul(x, self.dense1) + self.bias1)
            output = tf.matmul(hidden, self.dense2) + self.bias2
            return output
    
    model = CheckpointModel()
    
    # Create checkpoint manager
    checkpoint_dir = "./checkpoints"
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=3
    )
    
    print("=== Checkpoint Example ===")
    
    # Simulate training and saving checkpoints
    for epoch in range(5):
        # Simulate training step
        dummy_loss = tf.random.uniform(())
        model.global_step.assign_add(1)
        
        print(f"Epoch {epoch}, Loss: {dummy_loss:.4f}, Step: {model.global_step.numpy()}")
        
        # Save checkpoint every 2 epochs
        if epoch % 2 == 0:
            save_path = checkpoint_manager.save()
            print(f"  Checkpoint saved: {save_path}")
    
    # List available checkpoints
    print(f"\nAvailable checkpoints: {checkpoint_manager.checkpoints}")
    
    # Load latest checkpoint
    print("\n=== Loading Checkpoint ===")
    
    # Create new model instance
    new_model = CheckpointModel()
    new_checkpoint = tf.train.Checkpoint(model=new_model)
    
    # Show values before loading
    print(f"Before loading - Global step: {new_model.global_step.numpy()}")
    print(f"Before loading - Dense1 weights sum: {tf.reduce_sum(new_model.dense1).numpy():.4f}")
    
    # Load checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    new_checkpoint.restore(latest_checkpoint)
    
    print(f"Loaded checkpoint: {latest_checkpoint}")
    print(f"After loading - Global step: {new_model.global_step.numpy()}")
    print(f"After loading - Dense1 weights sum: {tf.reduce_sum(new_model.dense1).numpy():.4f}")
    
    # Manual checkpoint file inspection
    print("\n=== Checkpoint File Structure ===")
    checkpoint_files = tf.io.gfile.glob(f"{checkpoint_dir}/*")
    for file_path in sorted(checkpoint_files):
        file_name = os.path.basename(file_path)
        print(f"  {file_name}")
    
    return checkpoint_manager

checkpoint_mgr = checkpoint_example()
```

#### Advanced Checkpoint Features

```python
# Advanced checkpoint operations
def advanced_checkpoint_operations():
    """Advanced checkpoint features"""
    
    # Model with optimizer state
    class TrainingModel(tf.Module):
        def __init__(self):
            self.weights = tf.Variable(tf.random.normal([100, 50]), name="weights")
            self.bias = tf.Variable(tf.zeros([50]), name="bias")
            self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        
        @tf.function
        def train_step(self, x, y):
            with tf.GradientTape() as tape:
                predictions = tf.matmul(x, self.weights) + self.bias
                loss = tf.reduce_mean(tf.square(y - predictions))
            
            gradients = tape.gradient(loss, [self.weights, self.bias])
            self.optimizer.apply_gradients(zip(gradients, [self.weights, self.bias]))
            return loss
    
    model = TrainingModel()
    
    # Checkpoint with optimizer state
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=model.optimizer,
        step=tf.Variable(0, dtype=tf.int64)
    )
    
    # Train for a few steps
    print("=== Training with Checkpoints ===")
    x_train = tf.random.normal([32, 100])
    y_train = tf.random.normal([32, 50])
    
    for step in range(10):
        loss = model.train_step(x_train, y_train)
        checkpoint.step.assign_add(1)
        
        if step % 3 == 0:
            save_path = checkpoint.save(f"./training_checkpoints/step-{step}")
            print(f"Step {step}, Loss: {loss:.4f}, Saved: {save_path}")
    
    # Checkpoint inspection
    print("\n=== Checkpoint Variables ===")
    reader = tf.train.load_checkpoint("./training_checkpoints")
    
    # List all variables in checkpoint
    var_names = reader.get_variable_to_shape_map().keys()
    print("Variables in checkpoint:")
    for var_name in sorted(var_names):
        shape = reader.get_variable_to_shape_map()[var_name]
        dtype = reader.get_variable_to_dtype_map()[var_name].name
        print(f"  {var_name}: {shape} ({dtype})")
    
    # Read specific variable
    weights_value = reader.get_tensor("model/weights/.ATTRIBUTES/VARIABLE_VALUE")
    print(f"\nWeights shape: {weights_value.shape}")
    print(f"Weights mean: {np.mean(weights_value):.6f}")
    
    return checkpoint

advanced_checkpoint = advanced_checkpoint_operations()

## Learning Objectives

By the end of this section, you should be able to:

### Core Understanding
- **Explain the evolution** from TensorFlow 1.x to 2.x and choose appropriate approaches
- **Understand computation graphs** and their role in optimization and performance
- **Work with eager execution** for development and @tf.function for production
- **Navigate the TensorFlow ecosystem** and choose appropriate tools for different tasks

### Practical Skills
- **Manipulate tensors** efficiently with proper understanding of shapes, dtypes, and operations
- **Create and manage variables** for neural network parameters
- **Implement automatic differentiation** using GradientTape for custom training loops
- **Design custom loss functions** and metrics for specific problem domains
- **Build and execute** computation graphs using both high-level and low-level APIs

### Production Deployment
- **Use TensorFlow C++ API** for high-performance inference and integration
- **Build and link** TensorFlow C++ applications properly
- **Load and run SavedModels** in both Python and C++ environments
- **Understand model serialization** formats and their appropriate use cases
- **Work with checkpoints** for training resumption and model versioning

### Self-Assessment Checklist

Before proceeding to advanced topics, ensure you can:

□ Create a simple neural network using tf.Module and train it with custom training loops  
□ Convert between eager execution and graph mode (@tf.function)  
□ Implement custom gradients and higher-order derivatives  
□ Save and load models in SavedModel format with multiple signatures  
□ Build a TensorFlow C++ application that loads and runs a SavedModel  
□ Debug tensor shapes and data flow issues in complex models  
□ Optimize model performance using TensorFlow profiling tools  
□ Create custom losses, metrics, and preprocessing functions  

### Practical Exercises

**Exercise 1: Basic Tensor Operations**
```python
# TODO: Complete this tensor manipulation exercise
import tensorflow as tf

def tensor_exercise():
    # 1. Create a 4D tensor representing a batch of RGB images (batch=32, height=224, width=224, channels=3)
    images = # Your code here
    
    # 2. Calculate the mean pixel value across all images
    mean_pixel = # Your code here
    
    # 3. Normalize the images to have zero mean and unit variance
    normalized_images = # Your code here
    
    # 4. Reshape to prepare for a dense layer (flatten spatial dimensions)
    flattened = # Your code here
    
    return images, mean_pixel, normalized_images, flattened

# Test your implementation
images, mean_pixel, normalized, flattened = tensor_exercise()
print(f"Original shape: {images.shape}")
print(f"Mean pixel value: {mean_pixel}")
print(f"Normalized mean: {tf.reduce_mean(normalized)}")
print(f"Flattened shape: {flattened.shape}")
```

**Exercise 2: Custom Training Loop**
```python
# TODO: Implement a complete custom training loop
import tensorflow as tf

class LinearRegression(tf.Module):
    def __init__(self, input_dim):
        # Initialize weights and bias
        self.w = # Your code here
        self.b = # Your code here
    
    def __call__(self, x):
        # Forward pass
        return # Your code here

def train_model():
    # Generate synthetic data
    true_w = tf.constant([[2.0], [-1.5], [0.8]])
    true_b = tf.constant([1.0])
    
    # Your training implementation here
    model = LinearRegression(3)
    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    
    # Training loop
    for epoch in range(100):
        with tf.GradientTape() as tape:
            # Compute loss
            loss = # Your code here
        
        # Apply gradients
        # Your code here
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return model

trained_model = train_model()
```

**Exercise 3: SavedModel with Multiple Signatures**
```python
# TODO: Create a model with preprocessing and multiple serving signatures
class ImageClassifier(tf.Module):
    def __init__(self, num_classes=10):
        # Define your model architecture
        pass
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.uint8)])
    def serve_images(self, images):
        # Preprocess uint8 images and classify
        # Your code here
        pass
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
    def serve_features(self, features):
        # Classify pre-processed features
        # Your code here
        pass

# Save with multiple signatures and test loading
```

**Exercise 4: TensorFlow C++ Integration**
```cpp
// TODO: Complete this C++ program to load and run a SavedModel
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/framework/tensor.h>

int main() {
    // Load SavedModel
    tensorflow::SavedModelBundle model_bundle;
    tensorflow::Status status = // Your code here
    
    if (!status.ok()) {
        std::cerr << "Failed to load model: " << status.ToString() << std::endl;
        return -1;
    }
    
    // Prepare input tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, 
                                   tensorflow::TensorShape({1, 784}));
    
    // Fill input with data
    // Your code here
    
    // Run inference
    std::vector<tensorflow::Tensor> outputs;
    // Your code here
    
    // Print results
    // Your code here
    
    return 0;
}
```

## Study Materials

### Primary Resources
- **Official Documentation:** [TensorFlow Architecture Guide](https://www.tensorflow.org/guide)
- **Book:** "Hands-On Machine Learning" by Aurélien Géron (Chapters 10-12)
- **Book:** "TensorFlow for Deep Learning" by Bharath Ramsundar and Reza Bosagh Zadeh
- **Course:** TensorFlow Developer Certificate Program

### Advanced Resources
- **C++ API Documentation:** [TensorFlow C++ API Reference](https://www.tensorflow.org/api_docs/cc)
- **Research Papers:** 
  - "TensorFlow: A System for Large-Scale Machine Learning" (OSDI 2016)
  - "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems" (2015)
- **Blog Series:** TensorFlow Engineering Blog - Architecture Deep Dives

### Video Resources
- **YouTube:** TensorFlow Channel - "TensorFlow Explained" series
- **Coursera:** Deep Learning Specialization (Andrew Ng) - TensorFlow courses
- **edX:** MIT Introduction to Deep Learning - TensorFlow labs

### Hands-on Labs and Projects

**Lab 1: Performance Optimization**
- Compare eager vs graph execution performance
- Profile model execution with TensorBoard
- Optimize memory usage and computation

**Lab 2: Custom Operations**
- Implement custom TensorFlow operations in C++
- Create custom loss functions and metrics
- Build custom data pipelines

**Lab 3: Production Deployment**
- Deploy models using TensorFlow Serving
- Convert models to TensorFlow Lite
- Integrate TensorFlow C++ in existing applications

**Lab 4: Advanced Model Serialization**
- Work with complex SavedModel structures
- Implement model versioning strategies
- Handle model migration between TensorFlow versions

### Practice Projects

**Project 1: Multi-Modal Model**
```python
# Build a model that handles both images and text
class MultiModalModel(tf.Module):
    def __init__(self):
        # Image processing branch
        self.image_encoder = # Your implementation
        
        # Text processing branch  
        self.text_encoder = # Your implementation
        
        # Fusion layer
        self.fusion_layer = # Your implementation
    
    @tf.function
    def __call__(self, images, text):
        # Implement multi-modal fusion
        pass
```

**Project 2: Custom Training Framework**
```python
# Build a training framework with:
# - Automatic checkpointing
# - Learning rate scheduling
# - Early stopping
# - Mixed precision training
# - Distributed training support

class TrainingFramework:
    def __init__(self, model, optimizer, loss_fn):
        # Your implementation
        pass
    
    def train(self, train_dataset, val_dataset, epochs):
        # Your implementation
        pass
```

**Project 3: Model Analysis Tools**
```python
# Create tools for:
# - Model complexity analysis
# - Parameter counting
# - FLOP estimation
# - Memory profiling
# - Visualization of computation graphs

def analyze_model(model, input_shape):
    # Your implementation
    pass
```

### Development Environment Setup

**Python Environment:**
```bash
# Create virtual environment
python -m venv tf_env
source tf_env/bin/activate  # Linux/Mac
# tf_env\Scripts\activate  # Windows

# Install TensorFlow and dependencies
pip install tensorflow==2.13.0
pip install tensorboard
pip install tensorflow-datasets
pip install tensorflow-hub
pip install tensorflow-probability

# Development tools
pip install jupyter
pip install matplotlib
pip install pandas
pip install scikit-learn
```

**C++ Development:**
```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    python3-dev \
    python3-pip

# Install Bazel
wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-installer-linux-x86_64.sh
chmod +x bazel-5.3.0-installer-linux-x86_64.sh
./bazel-5.3.0-installer-linux-x86_64.sh --user

# Clone and build TensorFlow
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
bazel build //tensorflow:libtensorflow_cc.so
```

**Verification Scripts:**
```python
# Verify TensorFlow installation
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"Eager execution: {tf.executing_eagerly()}")

# Test basic functionality
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0, 6.0])
z = tf.add(x, y)
print(f"Basic operation result: {z}")

# Test gradient computation
with tf.GradientTape() as tape:
    tape.watch(x)
    loss = tf.reduce_sum(tf.square(x))
gradient = tape.gradient(loss, x)
print(f"Gradient: {gradient}")
```

### Assessment and Certification

**Knowledge Check Questions:**
1. What are the key differences between TensorFlow 1.x and 2.x execution models?
2. When should you use @tf.function vs eager execution?
3. How does automatic differentiation work in TensorFlow?
4. What are the components of a SavedModel and their purposes?
5. How do you optimize TensorFlow models for production deployment?

**Practical Assessments:**
- Build a complete training pipeline with custom components
- Implement model serving in both Python and C++
- Debug performance issues in complex models
- Create reproducible model versioning workflow

