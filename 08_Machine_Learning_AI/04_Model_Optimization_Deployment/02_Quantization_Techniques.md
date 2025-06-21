# Quantization Techniques

## Topics
- Precision formats (FP32, FP16, INT8, INT4)
- Post-training quantization
- Quantization-aware training
- Implementing quantized models

### Example: Post-training Quantization (TensorFlow)
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```
