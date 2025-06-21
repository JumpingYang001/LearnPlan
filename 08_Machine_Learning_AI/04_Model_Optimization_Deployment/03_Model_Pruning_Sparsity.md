# Model Pruning and Sparsity

## Topics
- Weight and activation pruning
- Structured vs. unstructured sparsity
- Magnitude-based and importance-based pruning
- Implementing pruned models

### Example: Pruning (PyTorch)
```python
import torch.nn.utils.prune as prune
prune.l1_unstructured(model.layer, name='weight', amount=0.5)
```
