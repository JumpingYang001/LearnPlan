# Modern ML Model Architectures

## Transformer Models
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Encoder-decoder architecture

## BERT and Variants
- Bidirectional training
- Masked language modeling
- Fine-tuning approaches
- Distilled versions

## GPT Models
- Autoregressive generation
- Scaling properties
- In-context learning
- Prompt engineering

## LLAMA and Open Source LLMs
- Architecture details
- Training methodology
- Fine-tuning approaches
- Deployment considerations

## Multi-Modal Models
- Text-image models
- Cross-modal attention
- Joint embeddings
- Generative capabilities

### Example: Transformer Block (PyTorch)
```python
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out
```

### Example: Use HuggingFace Transformers (Python)
```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
```
        return self.attn(x, x, x)[0]
```
