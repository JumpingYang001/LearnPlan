# Hardware Acceleration Fundamentals

## Topics
- Need for hardware acceleration in ML
- Accelerator types (GPU, TPU, FPGA, ASIC)
- Compute vs. memory-bound operations
- Basic parallelism concepts for ML workloads

### Example: Check CUDA Devices (Python)
```python
import torch
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
```
