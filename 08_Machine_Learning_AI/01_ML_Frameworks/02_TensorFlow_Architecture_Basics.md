# TensorFlow Architecture and Basics

## TensorFlow Overview
- Evolution (TF 1.x vs. TF 2.x)
- Computation graph concept
- Eager execution
- TensorFlow ecosystem

## Core Components
- Tensors and operations
- Variables and constants
- Automatic differentiation
- Metrics and losses

## TensorFlow C++ API
- TensorFlow C++ architecture
- Building and linking
- Session management
- Graph definition and execution
- Loading and running models

## TensorFlow Model Representation
- SavedModel format
- Protocol buffers
- GraphDef and MetaGraphDef
- Checkpoint files


### Example: Load Model in Python (TensorFlow)
```python
import tensorflow as tf
model = tf.saved_model.load('saved_model_dir')
infer = model.signatures['serving_default']
# result = infer(tf.constant(input_data))
```

### Example: Load Model in C++
```cpp
#include <tensorflow/cc/saved_model/loader.h>
// ... Load a SavedModel using TensorFlow C++ API ...
```
