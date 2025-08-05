# BERT Architecture and Applications

*Duration: 2 weeks*

## Overview

BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP by learning bidirectional context representations. Unlike previous models that processed text left-to-right or right-to-left, BERT learns from both directions simultaneously.

## 1. BERT's Bidirectional Transformer Approach

### Traditional vs. Bidirectional Context

**Traditional Approach (GPT-style):**
```
"The cat sat on the [MASK]"
   ←←←←←←←←←← (unidirectional)
```

**BERT's Bidirectional Approach:**
```
"The cat sat on the [MASK]"
   ←→←→←→←→←→ (bidirectional)
```

### Architecture Details

BERT uses only the **Encoder** part of the Transformer architecture:

```
Input: "The cat sat on the mat"
       ↓
┌─────────────────────────────┐
│    Token Embeddings         │
│  + Segment Embeddings       │
│  + Position Embeddings      │
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│   Transformer Encoder       │
│   (12 or 24 layers)         │
│                             │
│  ┌─ Multi-Head Attention ─┐ │
│  │  Add & Norm             │ │
│  │  Feed Forward           │ │
│  │  Add & Norm             │ │
│  └─────────────────────────┘ │
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│   Contextual Embeddings     │
└─────────────────────────────┘
```

### Key Components

**1. Input Representation:**
```python
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentence
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

print("Input IDs:", inputs['input_ids'])
print("Attention Mask:", inputs['attention_mask'])
print("Token Type IDs:", inputs['token_type_ids'])

# Decode to see special tokens
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
```

**Output:**
```
Tokens: ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '[SEP]']
```

**2. Three Types of Embeddings:**
```python
# Token embeddings: vocabulary mappings
# Segment embeddings: distinguish sentence A from B
# Position embeddings: indicate word position
total_embedding = token_emb + segment_emb + position_emb
```

## 2. Masked Language Modeling Pre-training

### MLM Objective

BERT randomly masks 15% of input tokens and learns to predict them using bidirectional context.

**Masking Strategy:**
- 80%: Replace with [MASK] token
- 10%: Replace with random token  
- 10%: Keep original token

**Example Implementation:**
```python
import torch
import random
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def mask_tokens(text, mask_prob=0.15):
    tokens = tokenizer.tokenize(text)
    masked_tokens = []
    labels = []
    
    for token in tokens:
        if random.random() < mask_prob:
            rand = random.random()
            if rand < 0.8:
                masked_tokens.append('[MASK]')
            elif rand < 0.9:
                masked_tokens.append(tokenizer.tokenize(random.choice(tokenizer.vocab))[0])
            else:
                masked_tokens.append(token)
            labels.append(tokenizer.convert_tokens_to_ids(token))
        else:
            masked_tokens.append(token)
            labels.append(-100)  # Ignore in loss calculation
    
    return masked_tokens, labels

# Example
text = "The cat sat on the mat"
masked_tokens, labels = mask_tokens(text)
print("Original:", tokenizer.tokenize(text))
print("Masked:", masked_tokens)
```

### Next Sentence Prediction (NSP)

BERT also learns to predict if sentence B follows sentence A:

```python
# Positive example
sentence_a = "The cat sat on the mat"
sentence_b = "It was very comfortable"
# Label: IsNext (1)

# Negative example  
sentence_a = "The cat sat on the mat"
sentence_b = "The weather is nice today"
# Label: NotNext (0)

# Encoding for NSP
inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
```

**Output:**
```
Tokens: ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '[SEP]', 
         'it', 'was', 'very', 'comfortable', '[SEP]']
```

## 3. Fine-tuning for Downstream Tasks

### Classification Tasks

**Sentiment Analysis Example:**
```python
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Sample data
texts = [
    "I love this movie!",
    "This film is terrible.",
    "Great acting and plot.",
    "Waste of time."
]
labels = [1, 0, 1, 0]  # 1: positive, 0: negative

# Create dataset
dataset = SentimentDataset(texts, labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Fine-tune
trainer.train()
```

### Named Entity Recognition (NER)

```python
from transformers import BertForTokenClassification
import torch

# NER tags: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
num_labels = 7
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

def predict_entities(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Map predictions to labels
    label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}
    
    for token, pred in zip(tokens, predictions[0]):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            print(f"{token}: {label_map[pred.item()]}")

# Example usage
text = "John works at Google in California"
predict_entities(text)
```

### Question Answering

```python
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

def answer_question(context, question):
    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)
    
    # Extract answer
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    answer_tokens = tokens[start_idx:end_idx+1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer

# Example
context = "BERT is a transformer-based model developed by Google. It uses bidirectional training."
question = "Who developed BERT?"
answer = answer_question(context, question)
print(f"Answer: {answer}")
```

## 4. Applications Using BERT Models

### Real-World Applications

**1. Search Engines:**
```python
# Google uses BERT for search query understanding
# Example: "parking on a hill with no curb" vs "parking on a hill with no curb"
# BERT understands the nuanced difference
```

**2. Content Moderation:**
```python
from transformers import pipeline

classifier = pipeline("text-classification", 
                     model="unitary/toxic-bert")

texts = [
    "This is a great post!",
    "I hate this content"
]

for text in texts:
    result = classifier(text)
    print(f"Text: {text}")
    print(f"Classification: {result}")
```

**3. Document Similarity:**
```python
from transformers import BertModel
import torch.nn.functional as F

model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :]

def calculate_similarity(text1, text2):
    emb1 = get_sentence_embedding(text1)
    emb2 = get_sentence_embedding(text2)
    
    similarity = F.cosine_similarity(emb1, emb2)
    return similarity.item()

# Example
text1 = "The cat sat on the mat"
text2 = "A feline rested on the rug"
similarity = calculate_similarity(text1, text2)
print(f"Similarity: {similarity:.4f}")
```

**4. Text Summarization:**
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based 
machine learning technique for natural language processing pre-training developed by Google. 
BERT makes use of Transformer, an attention mechanism that learns contextual relations 
between words in a text. Transformer includes two separate mechanisms — an encoder that 
reads the text input and a decoder that produces a prediction for the task. Since BERT's 
goal is to generate a language model, only the encoder mechanism is necessary.
"""

summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print("Summary:", summary[0]['summary_text'])
```

## 5. BERT Variants and Evolution

### Popular BERT Variants

**1. RoBERTa (Robustly Optimized BERT):**
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Key improvements:
# - Removed NSP task
# - Dynamic masking
# - Larger batch sizes
# - More training data
```

**2. DistilBERT (Distilled BERT):**
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# 60% smaller, 60% faster, 97% performance retention
```

**3. ALBERT (A Lite BERT):**
```python
from transformers import AlbertTokenizer, AlbertForSequenceClassification

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')

# Parameter sharing across layers
# Factorized embedding parameterization
```

## 6. Performance Benchmarks

### GLUE Benchmark Results

| Model | MNLI | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE |
|-------|------|-----|------|-------|------|-------|------|-----|
| BERT-base | 84.6 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 |
| BERT-large | 86.7 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 |

### Code to Evaluate on GLUE:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

def evaluate_on_glue_task(task_name):
    # Load dataset
    dataset = load_dataset("glue", task_name)
    
    # Load model
    model_name = f"textattack/bert-base-uncased-{task_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Evaluate on test set
    test_data = dataset['test']
    # Implementation details...
    
    return accuracy

# Example
accuracy = evaluate_on_glue_task('sst2')
print(f"SST-2 Accuracy: {accuracy}")
```

## 7. Learning Objectives

By the end of this section, you should be able to:
- **Explain** BERT's bidirectional approach and its advantages over unidirectional models
- **Implement** masked language modeling for pre-training
- **Fine-tune** BERT models for various downstream tasks (classification, NER, QA)
- **Apply** BERT to real-world applications like search and content moderation
- **Compare** different BERT variants and their trade-offs
- **Evaluate** BERT models using standard benchmarks

### Self-Assessment Checklist

□ Can explain the difference between BERT and GPT architectures  
□ Can implement MLM masking strategy  
□ Can fine-tune BERT for sentiment analysis  
□ Can use BERT for named entity recognition  
□ Can calculate document similarity using BERT embeddings  
□ Can choose appropriate BERT variant for specific use cases  
□ Can evaluate model performance on standard benchmarks  

## 8. Practical Exercises

**Exercise 1: Custom Classification**
```python
# TODO: Fine-tune BERT for spam email detection
# Dataset: emails with spam/not-spam labels
# Measure precision, recall, F1-score
```

**Exercise 2: Entity Extraction**
```python
# TODO: Build a NER system for social media posts
# Extract person names, organizations, locations
# Handle informal text and abbreviations
```

**Exercise 3: Semantic Search**
```python
# TODO: Create a document search system using BERT embeddings
# Index documents, query with natural language
# Rank results by semantic similarity
```

## 9. Study Materials

### Essential Reading
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [DistilBERT: a distilled version of BERT](https://arxiv.org/abs/1910.01108)

### Tutorials and Guides
- [HuggingFace BERT Tutorial](https://huggingface.co/docs/transformers/model_doc/bert)
- [Jay Alammar's Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [BERT Fine-tuning Tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)

### Tools and Libraries
```bash
pip install transformers datasets torch
pip install tokenizers
pip install evaluate  # for metrics
```
