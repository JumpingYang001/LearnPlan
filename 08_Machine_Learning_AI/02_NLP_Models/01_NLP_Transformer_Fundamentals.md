
# NLP and Transformer Fundamentals

## 1. Basic NLP Concepts and Challenges

Natural Language Processing (NLP) is a field at the intersection of computer science, artificial intelligence, and linguistics. It focuses on enabling computers to understand, interpret, and generate human language.

### Key Challenges in NLP
- **Ambiguity:** Words and sentences can have multiple meanings (e.g., "bank" as a financial institution or river bank).
- **Context:** Meaning often depends on context, which is hard for machines to capture.
- **Long-range dependencies:** Understanding a word may require information from far earlier in the text.
- **Data sparsity:** Many valid sentences are rare or unseen in training data.

**Example:**
> "He saw the man with the telescope."
Is the man holding the telescope, or is the observer using it?

---

## 2. Evolution from RNNs to Transformers

### Recurrent Neural Networks (RNNs)
- Process sequences step by step, maintaining a hidden state.
- Good for short sequences, but struggle with long-range dependencies due to vanishing gradients.

**RNN Limitation Example:**
```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)
inputs = torch.randn(1, 100, 10)  # Sequence of length 100
h0 = torch.zeros(1, 1, 20)
output, hn = rnn(inputs, h0)
# Information from the start of the sequence is hard to retain at the end
```

### Attention Mechanism
- Allows the model to focus on relevant parts of the input sequence, regardless of their position.
- Computes a weighted sum of all input positions for each output position.

### Transformers
- Introduced in "Attention is All You Need" (Vaswani et al., 2017)
- Rely entirely on attention mechanisms, removing recurrence and convolutions.
- Enable parallel processing of sequences and capture long-range dependencies efficiently.

---

## 3. Attention Mechanisms and Self-Attention

### What is Attention?
Given a query, attention computes a weighted sum over a set of key-value pairs:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Where:
- Q = Query matrix
- K = Key matrix
- V = Value matrix
- d_k = dimension of keys

### Self-Attention
- Each position in the sequence attends to all positions (including itself) to compute a new representation.

**Self-Attention Example (Python):**
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
print('Attention weights:', attn_weights)
print('Output:', output)
```

**Visualization:**
```
Input:  [The, cat, sat, on, mat]
	   |    |    |    |    |
	   v    v    v    v    v
Self-Attention computes how much each word should attend to every other word.
```

---

## 4. Transformer Architecture Fundamentals

### High-Level Structure

```
Input Embedding → Positional Encoding → [Encoder Blocks] → [Decoder Blocks] → Output
```

### Encoder Block
- Multi-Head Self-Attention
- Add & Norm
- Feed-Forward Network
- Add & Norm

### Decoder Block
- Masked Multi-Head Self-Attention
- Add & Norm
- Encoder-Decoder Attention
- Add & Norm
- Feed-Forward Network
- Add & Norm

**Diagram:**
```
┌────────────┐
│  Input     │
└─────┬──────┘
	│
	▼
┌────────────┐
│ Embedding  │
└─────┬──────┘
	│
	▼
┌────────────┐
│ Positional │
│ Encoding   │
└─────┬──────┘
	│
	▼
┌─────────────────────────────┐
│      Encoder Block(s)       │
└─────────────────────────────┘
	│
	▼
┌─────────────────────────────┐
│      Decoder Block(s)       │
└─────────────────────────────┘
	│
	▼
┌────────────┐
│  Output    │
└────────────┘
```

### Multi-Head Attention (MHA)
- Runs multiple self-attention operations in parallel ("heads")
- Each head learns to focus on different aspects of the sequence

**MHA Example (PyTorch):**
```python
import torch.nn as nn
mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
Q = K = V = torch.rand(1, 5, 8)
output, attn_weights = mha(Q, K, V)
print('MHA output:', output)
print('MHA weights:', attn_weights)
```

### Feed-Forward Network (FFN)
- Applies two linear transformations with a ReLU activation in between

**FFN Example:**
```python
import torch.nn.functional as F
def ffn(x):
    return F.relu(torch.nn.Linear(8, 32)(x))
```

### Positional Encoding
- Since transformers have no recurrence, positional encoding injects information about word order

**Positional Encoding Example:**
```python
import torch
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
	  for i in range(0, d_model, 2):
		pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
		if i+1 < d_model:
		    pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
    return pe

print(positional_encoding(5, 8))
```

---

## 5. Learning Objectives

By the end of this section, you should be able to:
- Explain the limitations of RNNs and the motivation for transformers
- Describe the self-attention mechanism and compute it by hand or in code
- Sketch the high-level architecture of a transformer
- Implement a simple transformer block in PyTorch
- Understand the role of positional encoding
- Use multi-head attention in practice

---

## 6. Practice Questions & Exercises

**Conceptual:**
1. Why do RNNs struggle with long-range dependencies?
2. What problem does self-attention solve in NLP?
3. How does positional encoding work?
4. What is the benefit of multi-head attention?

**Technical:**
5. Write code to compute self-attention for a batch of sequences.
6. Modify the self-attention code to mask out future positions (for language modeling).
7. Implement a simple positional encoding function.

**Challenge:**
8. Build a minimal transformer encoder block in PyTorch.
9. Visualize the attention weights for a sample sentence.

---

## 7. Further Reading & Resources

- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch nn.Transformer documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

---

## 8. Development Environment Setup

**Required:**
```bash
pip install torch
```

**Optional:**
- Jupyter Notebook for interactive experimentation
- Matplotlib for visualizing attention weights

---

## 9. Hands-on Lab Ideas

- Implement a toy transformer for text classification
- Visualize attention weights for a sentence
- Compare transformer performance to RNN on a small dataset
