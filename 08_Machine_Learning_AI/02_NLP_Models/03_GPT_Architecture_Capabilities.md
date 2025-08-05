# GPT Architecture and Capabilities

*Duration: 2 weeks*

## Overview

GPT (Generative Pretrained Transformer) represents a paradigm shift in natural language processing, demonstrating that large-scale autoregressive language models can perform a wide variety of tasks through in-context learning and prompt engineering.

## 1. GPT's Autoregressive Transformer Design

### Autoregressive vs. Bidirectional

**BERT (Bidirectional):**
```
[CLS] The cat sat on the [MASK] [SEP]
  ←→  ←→  ←→  ←→  ←→   ←→    ←→
```

**GPT (Autoregressive/Causal):**
```
The cat sat on the mat
→   →   →   →   →   →
```

GPT can only see previous tokens when predicting the next token, making it naturally suited for text generation.

### Architecture Differences

**GPT vs BERT Architecture:**

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│         BERT                │    │         GPT                 │
├─────────────────────────────┤    ├─────────────────────────────┤
│   [CLS] + Sentence + [SEP]  │    │    Start + Text Sequence    │
│           ↓                 │    │           ↓                 │
│    Bidirectional Encoder    │    │  Causal/Masked Decoder      │
│    (can see all tokens)     │    │  (can only see previous)    │
│           ↓                 │    │           ↓                 │
│   Classification Head       │    │   Language Modeling Head    │
└─────────────────────────────┘    └─────────────────────────────┘
```

### Masked Self-Attention in GPT

```python
import torch
import torch.nn.functional as F
import numpy as np

def create_causal_mask(seq_len):
    """Create causal mask to prevent attention to future positions"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = torch.from_numpy(mask) == 0
    return mask

def causal_self_attention(Q, K, V):
    """Self-attention with causal masking"""
    d_k = Q.size(-1)
    seq_len = Q.size(-2)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply causal mask
    mask = create_causal_mask(seq_len)
    scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Example usage
seq_len, d_model = 5, 8
Q = K = V = torch.randn(1, seq_len, d_model)
output, attn_weights = causal_self_attention(Q, K, V)

print("Attention weights (note lower triangular structure):")
print(attn_weights[0].detach().numpy())
```

### GPT Block Structure

```python
import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head causal self-attention
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT uses GELU instead of ReLU
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (GPT uses pre-norm)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, causal_mask=None):
        # Pre-norm + self-attention + residual
        normed = self.ln1(x)
        attn_out, _ = self.self_attention(
            normed, normed, normed, 
            attn_mask=causal_mask, 
            is_causal=True
        )
        x = x + self.dropout(attn_out)
        
        # Pre-norm + FFN + residual
        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x

# Example usage
block = GPTBlock(d_model=512, n_heads=8, d_ff=2048)
input_seq = torch.randn(1, 10, 512)  # batch_size=1, seq_len=10, d_model=512
output = block(input_seq)
print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
```

## 2. Scaling Laws and Model Sizes

### GPT Model Evolution

| Model | Parameters | Layers | d_model | n_heads | Release |
|-------|------------|--------|---------|---------|---------|
| GPT-1 | 117M | 12 | 768 | 12 | 2018 |
| GPT-2 Small | 124M | 12 | 768 | 12 | 2019 |
| GPT-2 Medium | 355M | 24 | 1024 | 16 | 2019 |
| GPT-2 Large | 774M | 36 | 1280 | 20 | 2019 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 | 2019 |
| GPT-3 | 175B | 96 | 12288 | 96 | 2020 |
| GPT-4 | ~1.7T* | ~120* | ~18432* | ~128* | 2023 |

*Estimated values

### Scaling Laws

**Key Findings from OpenAI's Scaling Laws:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Illustrative scaling law relationship
def compute_loss(N, D, C):
    """
    N: Number of parameters
    D: Dataset size  
    C: Compute budget
    """
    # Simplified version of scaling laws
    alpha_N, alpha_D, alpha_C = 0.076, 0.095, 0.057
    N_c, D_c, C_c = 8.8e6, 5.4e6, 3.1e8
    
    loss = 1.69 + (N_c/N)**alpha_N + (D_c/D)**alpha_D + (C_c/C)**alpha_C
    return loss

# Generate scaling curves
params = np.logspace(6, 11, 50)  # 1M to 100B parameters
losses = [compute_loss(N, 1e10, 1e20) for N in params]

plt.figure(figsize=(10, 6))
plt.loglog(params, losses)
plt.xlabel('Number of Parameters')
plt.ylabel('Test Loss')
plt.title('GPT Scaling Laws: Loss vs Model Size')
plt.grid(True)
plt.show()

# Key insights:
print("Scaling Law Insights:")
print("1. Performance improves predictably with scale")
print("2. Larger models are more sample-efficient")
print("3. Compute-optimal training requires balanced scaling")
```

### Memory and Compute Requirements

```python
def estimate_gpt_requirements(n_params, seq_len=2048, batch_size=1):
    """Estimate memory and compute for GPT inference"""
    
    # Memory breakdown (in GB)
    model_memory = n_params * 2 / 1e9  # FP16 weights
    activation_memory = (24 * n_params * seq_len * batch_size) / 1e9  # Rough estimate
    kv_cache_memory = (2 * 32 * seq_len * 4096 * batch_size) / 1e9  # For large models
    
    total_memory = model_memory + activation_memory + kv_cache_memory
    
    # Compute (FLOPs per token)
    flops_per_token = 6 * n_params  # Forward pass estimate
    
    print(f"Model with {n_params/1e9:.1f}B parameters:")
    print(f"  Model weights: {model_memory:.1f} GB")
    print(f"  Activations: {activation_memory:.1f} GB") 
    print(f"  KV cache: {kv_cache_memory:.1f} GB")
    print(f"  Total memory: {total_memory:.1f} GB")
    print(f"  FLOPs per token: {flops_per_token/1e9:.1f}B")
    print()

# Examples
estimate_gpt_requirements(125e6)   # GPT-2 Small
estimate_gpt_requirements(1.5e9)   # GPT-2 XL  
estimate_gpt_requirements(175e9)   # GPT-3
```

## 3. Prompt Engineering and Few-Shot Learning

### Zero-Shot Learning

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Zero-shot task examples
prompts = [
    "Translate English to French: Hello, how are you?",
    "Sentiment (positive/negative): I love this movie!",
    "Summary: The quick brown fox jumps over the lazy dog. This is a famous pangram.",
]

for prompt in prompts:
    result = generate_text(prompt, max_length=50)
    print(f"Prompt: {prompt}")
    print(f"Output: {result}")
    print("-" * 50)
```

### Few-Shot Learning Examples

```python
def few_shot_classification():
    """Demonstrate few-shot sentiment classification"""
    prompt = """
    Text: I love this product! It's amazing.
    Sentiment: Positive
    
    Text: This is the worst thing I've ever bought.
    Sentiment: Negative
    
    Text: The product is okay, nothing special.
    Sentiment: Neutral
    
    Text: This exceeded all my expectations!
    Sentiment:"""
    
    result = generate_text(prompt, max_length=len(prompt.split()) + 5)
    print("Few-shot sentiment classification:")
    print(result)

def few_shot_math():
    """Demonstrate few-shot arithmetic reasoning"""
    prompt = """
    Q: What is 15 + 27?
    A: 42
    
    Q: What is 8 × 9?
    A: 72
    
    Q: What is 156 - 89?
    A: 67
    
    Q: What is 24 ÷ 6?
    A:"""
    
    result = generate_text(prompt, max_length=len(prompt.split()) + 5)
    print("Few-shot arithmetic:")
    print(result)

few_shot_classification()
few_shot_math()
```

### Advanced Prompt Engineering Techniques

```python
class PromptTemplate:
    """Template class for structured prompts"""
    
    def __init__(self, template):
        self.template = template
    
    def format(self, **kwargs):
        return self.template.format(**kwargs)

# Chain-of-thought prompting
cot_template = PromptTemplate("""
Question: {question}

Let me think step by step:
1. First, I need to identify what the question is asking
2. Then, I'll break down the problem into smaller parts
3. Finally, I'll solve each part and combine the results

Step-by-step solution:
""")

# Role-based prompting
role_template = PromptTemplate("""
You are {role}. {context}

User: {user_input}

{role}:
""")

# Examples
math_prompt = cot_template.format(
    question="If a train travels 60 mph for 2.5 hours, how far does it go?"
)

expert_prompt = role_template.format(
    role="a helpful AI assistant specializing in machine learning",
    context="You always provide accurate, well-explained answers with examples.",
    user_input="Explain the difference between supervised and unsupervised learning"
)

print("Chain-of-thought prompt:")
print(math_prompt)
print("\nRole-based prompt:")
print(expert_prompt)
```

## 4. Applications Using GPT Models

### Text Generation Applications

**1. Creative Writing Assistant:**
```python
def creative_writing_assistant(genre, characters, setting):
    prompt = f"""
    Write a {genre} story with the following elements:
    Characters: {characters}
    Setting: {setting}
    
    Story:
    """
    
    story = generate_text(prompt, max_length=200, temperature=0.8)
    return story

# Example
story = creative_writing_assistant(
    genre="science fiction",
    characters="a robot and a human scientist",
    setting="space station orbiting Mars"
)
print(story)
```

**2. Code Generation:**
```python
def generate_code(description, language="Python"):
    prompt = f"""
    # Task: {description}
    # Language: {language}
    
    def solution():
        # Implementation:
    """
    
    code = generate_text(prompt, max_length=150, temperature=0.3)
    return code

# Example
code = generate_code("Create a function to find the factorial of a number")
print(code)
```

**3. Chatbot Implementation:**
```python
class GPTChatbot:
    def __init__(self, personality="helpful assistant"):
        self.personality = personality
        self.conversation_history = []
        
    def chat(self, user_input):
        # Build conversation context
        context = f"You are a {self.personality}.\n\n"
        
        # Add conversation history
        for exchange in self.conversation_history[-5:]:  # Last 5 exchanges
            context += f"Human: {exchange['human']}\nAssistant: {exchange['assistant']}\n\n"
        
        # Add current input
        context += f"Human: {user_input}\nAssistant:"
        
        # Generate response
        response = generate_text(context, max_length=len(context.split()) + 50, temperature=0.7)
        assistant_response = response.split("Assistant:")[-1].strip()
        
        # Update history
        self.conversation_history.append({
            'human': user_input,
            'assistant': assistant_response
        })
        
        return assistant_response

# Example usage
chatbot = GPTChatbot("friendly and knowledgeable AI tutor")
print(chatbot.chat("What is machine learning?"))
print(chatbot.chat("Can you give me an example?"))
```

### Real-World Applications

**1. Content Generation for Marketing:**
```python
def generate_marketing_copy(product, target_audience, tone="professional"):
    prompt = f"""
    Product: {product}
    Target Audience: {target_audience}
    Tone: {tone}
    
    Create compelling marketing copy:
    
    Headline:
    """
    
    return generate_text(prompt, max_length=100, temperature=0.6)

copy = generate_marketing_copy(
    product="AI-powered fitness app",
    target_audience="busy professionals aged 25-40",
    tone="energetic and motivating"
)
print(copy)
```

**2. Educational Content Creation:**
```python
def create_lesson_plan(subject, grade_level, duration):
    prompt = f"""
    Subject: {subject}
    Grade Level: {grade_level}  
    Duration: {duration}
    
    Lesson Plan:
    
    Objective:
    Students will be able to...
    
    Materials:
    -
    
    Activities:
    1.
    """
    
    return generate_text(prompt, max_length=200, temperature=0.5)

lesson = create_lesson_plan(
    subject="Introduction to Photosynthesis",
    grade_level="5th grade",
    duration="45 minutes"
)
print(lesson)
```

**3. Data Analysis and Insights:**
```python
def analyze_data_trends(data_description, findings):
    prompt = f"""
    Data: {data_description}
    Key Findings: {findings}
    
    Analysis Report:
    
    Executive Summary:
    Based on the analysis of {data_description}, we found {findings}. 
    
    Detailed Analysis:
    """
    
    return generate_text(prompt, max_length=150, temperature=0.4)

analysis = analyze_data_trends(
    data_description="e-commerce sales data for Q3 2023",
    findings="20% increase in mobile purchases, 15% decrease in desktop sales"
)
print(analysis)
```

## 5. GPT Model Variants and Improvements

### GPT Model Family

**GPT-1 (2018):**
- 117M parameters
- Demonstrated unsupervised pre-training + supervised fine-tuning
- Showed transfer learning capabilities

**GPT-2 (2019):**
- Up to 1.5B parameters
- Demonstrated zero-shot task performance
- Initially withheld due to concerns about misuse

**GPT-3 (2020):**
- 175B parameters
- Strong few-shot learning capabilities
- API-based access model

**GPT-4 (2023):**
- Multimodal capabilities (text + images)
- Improved reasoning and safety
- Advanced prompt following

### Implementation Comparison

```python
# GPT-2 vs GPT-3 API usage examples

# GPT-2 (local)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

def gpt2_generate(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, temperature=0.7)
    return tokenizer.decode(outputs[0])

# GPT-3/4 (API - conceptual)
import openai  # Conceptual example

def gpt3_generate(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text

# Performance comparison
prompt = "The future of artificial intelligence is"
print("GPT-2:", gpt2_generate(prompt))
# print("GPT-3:", gpt3_generate(prompt))  # Requires API key
```

## 6. Performance Evaluation and Benchmarks

### Common Evaluation Metrics

```python
import numpy as np
from collections import Counter

def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity on text"""
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    
    perplexity = torch.exp(loss)
    return perplexity.item()

def calculate_bleu_score(reference, candidate):
    """Simple BLEU score calculation"""
    # Simplified implementation for demonstration
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    # Calculate n-gram precision for n=1,2,3,4
    precisions = []
    for n in range(1, 5):
        ref_ngrams = Counter([tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)])
        cand_ngrams = Counter([tuple(cand_words[i:i+n]) for i in range(len(cand_words)-n+1)])
        
        overlap = sum((ref_ngrams & cand_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(overlap / total)
    
    # Geometric mean
    if min(precisions) > 0:
        bleu = np.exp(np.mean(np.log(precisions)))
    else:
        bleu = 0
    
    return bleu

# Example usage
text = "The quick brown fox jumps over the lazy dog"
perplexity = calculate_perplexity(model, tokenizer, text)
print(f"Perplexity: {perplexity:.2f}")

reference = "The cat sat on the mat"
candidate = "A cat was sitting on the mat"
bleu = calculate_bleu_score(reference, candidate)
print(f"BLEU Score: {bleu:.3f}")
```

### Benchmark Results

**HellaSwag (Commonsense Reasoning):**
| Model | Accuracy |
|-------|----------|
| GPT-2 | 78.9% |
| GPT-3 | 95.3% |
| GPT-4 | 95.3% |

**MMLU (Massive Multitask Language Understanding):**
| Model | Average |
|-------|---------|
| GPT-3 | 43.9% |  
| GPT-4 | 86.4% |

## 7. Learning Objectives

By the end of this section, you should be able to:
- **Explain** the autoregressive architecture of GPT and how it differs from BERT
- **Understand** scaling laws and their implications for model development
- **Apply** prompt engineering techniques for various tasks
- **Implement** few-shot learning with GPT models
- **Build** applications using GPT for text generation and analysis
- **Evaluate** GPT model performance using appropriate metrics
- **Compare** different GPT variants and their capabilities

### Self-Assessment Checklist

□ Can explain causal/masked self-attention in GPT  
□ Can implement prompt templates for different tasks  
□ Can perform few-shot learning with examples  
□ Can build a simple chatbot using GPT  
□ Can calculate perplexity and other evaluation metrics  
□ Can choose appropriate GPT variant for specific use cases  
□ Can estimate compute and memory requirements for large models  

## 8. Practical Exercises

**Exercise 1: Prompt Engineering**
```python
# TODO: Create prompts for:
# 1. Email classification (spam/not spam)
# 2. Code debugging assistance  
# 3. Creative story continuation
# Test different prompt formats and compare results
```

**Exercise 2: Custom Text Generator**
```python
# TODO: Build a specialized text generator for:
# - Product descriptions
# - Social media posts
# - Technical documentation
# Include temperature and length controls
```

**Exercise 3: Evaluation Framework**
```python
# TODO: Create evaluation framework that measures:
# - Coherence of generated text
# - Task-specific accuracy
# - Computational efficiency
# Compare multiple GPT models
```

## 9. Study Materials

### Essential Papers
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### Tools and Libraries
```bash
pip install transformers torch
pip install openai  # For API access
pip install datasets  # For evaluation datasets
pip install nltk  # For text processing metrics
```

### Online Resources
- [GPT-3 Playground](https://platform.openai.com/playground)
- [Hugging Face GPT Models](https://huggingface.co/models?filter=gpt)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)