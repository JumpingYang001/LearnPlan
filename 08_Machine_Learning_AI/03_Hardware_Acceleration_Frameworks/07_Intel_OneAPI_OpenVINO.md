# Intel OneAPI and OpenVINO

## Topics
- Intel's acceleration approach
- DPC++ and SYCL programming
- OpenVINO for inference optimization
- Intel-optimized applications

### Example: OpenVINO Inference (Python)
```python
from openvino.runtime import Core
ie = Core()
model = ie.read_model(model='model.xml')
compiled_model = ie.compile_model(model=model, device_name='CPU')
# Run inference...
```
