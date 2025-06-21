# NLP and Transformer Fundamentals

## Topics
- Basic NLP concepts and challenges
- Evolution from RNNs to Transformers
- Attention mechanisms and self-attention
- Transformer architecture fundamentals

### Example: Self-Attention Calculation (Python)
```python
import torch
import torch.nn.functional as F

# Q, K, V: (batch, seq_len, d_model)
Q = torch.rand(1, 5, 8)
K = torch.rand(1, 5, 8)
V = torch.rand(1, 5, 8)
attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
attn_weights = F.softmax(attn_scores, dim=-1)
output = torch.matmul(attn_weights, V)
```
