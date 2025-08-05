# LLAMA and Open-Source LLMs

*Duration: 2 weeks*

## Overview

LLAMA (Large Language Model Meta AI) represents a breakthrough in open-source large language models, demonstrating that smaller, more efficient models can compete with much larger proprietary models through better training techniques and data quality.

## 1. LLAMA Architecture and Innovations

### Key Architectural Improvements

LLAMA incorporates several improvements over the standard Transformer architecture:

**1. RMSNorm instead of LayerNorm:**
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Benefits of RMSNorm:
# - Simpler computation (no mean subtraction)
# - Slightly faster
# - No bias parameter needed
# - More stable training
```

**2. SwiGLU Activation Function:**
```python
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """SwiGLU activation function used in LLAMA"""
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2, bias=False)
        self.w2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Split the linear transformation
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        
        # Apply SwiGLU: swish(x1) * x2
        swish_x1 = x1 * torch.sigmoid(x1)  # Swish activation
        gated = swish_x1 * x2  # Gating mechanism
        
        return self.w2(gated)

# SwiGLU provides better gradient flow and performance
```

**3. Rotary Positional Embeddings (RoPE):**
```python
import math

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        return cos_freqs, sin_freqs

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

## 2. Open-Source Alternatives to Proprietary Models

### LLAMA Model Family

**LLAMA 1 (2023):**
| Model | Parameters | Context Length | Training Tokens |
|-------|------------|----------------|-----------------|
| LLAMA-7B | 7B | 2048 | 1T |
| LLAMA-13B | 13B | 2048 | 1T |
| LLAMA-30B | 30B | 2048 | 1.4T |
| LLAMA-65B | 65B | 2048 | 1.4T |

**Loading and Using LLAMA Models:**
```python
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

def load_llama_model(model_name="meta-llama/Llama-2-7b-hf"):
    """Load LLAMA model with proper configuration"""
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def generate_with_llama(model, tokenizer, prompt, max_length=100):
    """Generate text using LLAMA model"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 3. Efficiency Improvements in LLAMA 2/3

### Training Efficiency Improvements

**1. Grouped Query Attention (GQA):**
```python
class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention for efficiency"""
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Repeat K and V for each group
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Standard attention computation...
        return output
```

**2. KV Cache Optimization:**
```python
class KVCache:
    """Key-Value cache for efficient inference"""
    def __init__(self, max_seq_len, n_heads, head_dim, dtype=torch.float16):
        self.k_cache = torch.zeros(1, n_heads, max_seq_len, head_dim, dtype=dtype)
        self.v_cache = torch.zeros(1, n_heads, max_seq_len, head_dim, dtype=dtype)
        self.cache_len = 0

    def update(self, k, v, start_pos):
        seq_len = k.size(2)
        self.k_cache[:, :, start_pos:start_pos + seq_len] = k
        self.v_cache[:, :, start_pos:start_pos + seq_len] = v
        self.cache_len = start_pos + seq_len

    def get(self, start_pos, seq_len):
        return (
            self.k_cache[:, :, :start_pos + seq_len],
            self.v_cache[:, :, :start_pos + seq_len]
        )
```

## 4. Applications Using LLAMA Models

### Fine-tuning for Specific Tasks

**1. Instruction Following:**
```python
from transformers import Trainer, TrainingArguments

class InstructionDataset:
    def __init__(self, instructions, responses, tokenizer, max_length=512):
        self.instructions = instructions
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        response = self.responses[idx]
        
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }
```

**2. Code Generation with Code LLAMA:**
```python
def generate_code(model, tokenizer, prompt, language="python"):
    """Generate code using Code LLAMA"""
    formatted_prompt = f"# Language: {language}\n# Task: {prompt}\n\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code.replace(formatted_prompt, "").strip()
```

## 5. Learning Objectives

By the end of this section, you should be able to:
- **Explain** LLAMA's architectural innovations and their benefits
- **Compare** open-source alternatives to proprietary models
- **Implement** efficiency improvements like GQA and KV caching
- **Fine-tune** LLAMA models for specific applications
- **Deploy** LLAMA models with optimization techniques
- **Evaluate** trade-offs between model size, speed, and performance

## 6. Practical Exercises

**Exercise 1: Model Comparison**
```python
# TODO: Compare LLAMA variants on standard benchmarks
# Measure performance, memory usage, and inference speed
```

**Exercise 2: Custom Fine-tuning**
```python
# TODO: Fine-tune LLAMA-7B for domain-specific task
# Choose from: medical QA, legal analysis, or code generation
```

**Exercise 3: Optimization Implementation**
```python
# TODO: Implement and benchmark optimization techniques
# Compare standard attention vs. grouped query attention
```

## 7. Study Materials

### Essential Papers
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950)

### Tools and Libraries
```bash
pip install transformers torch
pip install accelerate bitsandbytes  # For optimization
pip install datasets evaluate  # For training and evaluation
```
