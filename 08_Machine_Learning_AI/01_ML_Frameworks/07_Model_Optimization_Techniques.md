# Model Optimization Techniques

## Quantization
- Post-training quantization
- Quantization-aware training
- INT8/FP16 computation
- Symmetric vs. asymmetric quantization

## Pruning
- Weight pruning
- Structured vs. unstructured pruning
- Magnitude-based pruning
- Iterative pruning

## Knowledge Distillation
- Teacher-student models
- Distillation loss
- Feature distillation
- Implementation techniques

## Model Compression
- Weight sharing
- Low-rank factorization
- Huffman coding
- Tensor decomposition

### Example: Quantization (PyTorch)
```python
import torch.quantization as tq
model_fp32 = ...
model_int8 = tq.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)
```
