# Fine-tuning Transformer Models

*Duration: 3 weeks*

## Overview

Fine-tuning allows us to adapt pre-trained transformer models to specific tasks and domains. With the advent of large language models, parameter-efficient fine-tuning methods have become crucial for practical deployment while maintaining performance.

## 1. Parameter-Efficient Fine-Tuning Methods

### Traditional Fine-tuning vs. Parameter-Efficient Methods

**Traditional Full Fine-tuning:**
- Updates all model parameters
- Requires large memory and compute
- Risk of catastrophic forgetting
- Separate model copy for each task

**Parameter-Efficient Fine-tuning:**
- Updates only a small subset of parameters
- Maintains most of the original model
- Lower memory requirements
- Better generalization

### Comparison of Methods

| Method | Trainable Parameters | Memory Usage | Performance | Complexity |
|--------|---------------------|--------------|-------------|------------|
| Full Fine-tuning | 100% | High | Excellent | Low |
| LoRA | ~0.1-1% | Low | Excellent | Medium |
| Adapters | ~2-4% | Medium | Good | Medium |
| Prompt Tuning | ~0.01% | Very Low | Good | High |
| Prefix Tuning | ~0.1% | Low | Good | High |

## 2. LoRA (Low-Rank Adaptation)

### Theory Behind LoRA

LoRA assumes that weight updates during fine-tuning have a low intrinsic rank. Instead of updating the full weight matrix W, it learns a low-rank decomposition:

```
W_new = W_original + ΔW
ΔW = A × B
```

Where A ∈ R^(d×r) and B ∈ R^(r×k), with r << min(d,k)

### LoRA Implementation

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Scaling factor
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        # LoRA computation: x @ (A @ B) * scaling
        result = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return result

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, original_layer, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Add LoRA adaptation
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x):
        # Original output + LoRA adaptation
        return self.original_layer(x) + self.lora(x)

# Example: Apply LoRA to a transformer model
def apply_lora_to_model(model, rank=4, alpha=16, target_modules=None):
    """Apply LoRA to specified modules in a model"""
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'out_proj']
    
    lora_modules = {}
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                
                # Set the new module
                parent = model
                for attr in parent_name.split('.'):
                    if attr:
                        parent = getattr(parent, attr)
                setattr(parent, child_name, lora_layer)
                
                lora_modules[name] = lora_layer
    
    return model, lora_modules

# Count trainable parameters
def count_parameters(model):
    """Count trainable vs total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Percentage trainable: {100 * trainable / total:.2f}%")
    
    return trainable, total
```

### Practical LoRA Fine-tuning with PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import torch

def setup_lora_model(model_name, num_labels=2, task_type=TaskType.SEQ_CLS):
    """Setup model with LoRA configuration"""
    
    # Load base model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=task_type,
        r=16,  # rank
        lora_alpha=32,  # scaling parameter
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

# Example: Fine-tune BERT with LoRA for sentiment analysis
def finetune_sentiment_with_lora():
    """Complete example of LoRA fine-tuning for sentiment analysis"""
    
    # Setup model
    model, tokenizer = setup_lora_model("bert-base-uncased", num_labels=2)
    
    # Sample dataset
    from datasets import Dataset
    
    texts = [
        "I love this movie! It's fantastic.",
        "This film is terrible and boring.",
        "Great acting and wonderful story.",
        "Waste of time, very disappointing.",
        "Amazing cinematography and direction.",
        "Poor script and bad performances."
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
    
    dataset = Dataset.from_dict({"text": texts, "labels": labels})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora-sentiment-results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-4,  # Higher LR for LoRA
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Fine-tune
    trainer.train()
    
    return model, tokenizer

# Save and load LoRA adapters
def save_lora_adapters(model, save_path):
    """Save only the LoRA adapters"""
    model.save_pretrained(save_path)
    print(f"LoRA adapters saved to {save_path}")

def load_lora_adapters(base_model_name, adapter_path):
    """Load LoRA adapters onto base model"""
    from peft import PeftModel
    
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model
```

## 3. Prompt Tuning and Other Parameter-Efficient Methods

### Prompt Tuning

Prompt tuning prepends learnable "soft prompts" to the input while keeping the model frozen.

```python
import torch
import torch.nn as nn

class SoftPromptTuning(nn.Module):
    """Soft prompt tuning implementation"""
    def __init__(self, model, prompt_length=10, hidden_size=768):
        super().__init__()
        self.model = model
        self.prompt_length = prompt_length
        
        # Learnable soft prompts
        self.soft_prompt = nn.Parameter(
            torch.randn(1, prompt_length, hidden_size) * 0.02
        )
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)
        
        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Expand soft prompts for batch
        soft_prompts = self.soft_prompt.expand(batch_size, -1, -1)
        
        # Concatenate soft prompts with input embeddings
        inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.prompt_length, 
                device=attention_mask.device, 
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # Forward pass
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

# Example usage
def setup_prompt_tuning(model_name, prompt_length=10):
    """Setup model with prompt tuning"""
    from transformers import AutoModel
    
    base_model = AutoModel.from_pretrained(model_name)
    prompt_model = SoftPromptTuning(
        base_model, 
        prompt_length=prompt_length,
        hidden_size=base_model.config.hidden_size
    )
    
    # Count parameters
    trainable = sum(p.numel() for p in prompt_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in prompt_model.parameters())
    
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Percentage trainable: {100 * trainable / total:.4f}%")
    
    return prompt_model
```

### Adapter Layers

Adapters insert small bottleneck layers between transformer blocks.

```python
class AdapterLayer(nn.Module):
    """Adapter layer implementation"""
    def __init__(self, input_size, bottleneck_size=64, activation='gelu'):
        super().__init__()
        
        self.down_project = nn.Linear(input_size, bottleneck_size)
        self.up_project = nn.Linear(bottleneck_size, input_size)
        
        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Initialize near-zero for stable training
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # Bottleneck transformation
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        
        # Residual connection
        return residual + x

def add_adapters_to_transformer(model, bottleneck_size=64):
    """Add adapter layers to transformer blocks"""
    
    for name, module in model.named_modules():
        if 'layer' in name and hasattr(module, 'output'):
            # Add adapter after each transformer layer
            original_forward = module.forward
            adapter = AdapterLayer(
                module.output.dense.out_features, 
                bottleneck_size
            )
            
            def new_forward(self, hidden_states, *args, **kwargs):
                output = original_forward(hidden_states, *args, **kwargs)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    adapted = adapter(hidden_states)
                    return (adapted,) + output[1:]
                else:
                    return adapter(output)
            
            # Replace forward method
            module.forward = new_forward.__get__(module, module.__class__)
            
            # Freeze original parameters
            for param in module.parameters():
                param.requires_grad = False
            
            # Make adapter parameters trainable
            for param in adapter.parameters():
                param.requires_grad = True
    
    return model
```

## 4. Instruction Tuning and RLHF

### Instruction Tuning

Instruction tuning trains models to follow natural language instructions.

```python
class InstructionTuningDataset:
    """Dataset for instruction tuning"""
    def __init__(self, instructions, inputs, outputs, tokenizer, max_length=512):
        self.instructions = instructions
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        input_text = self.inputs[idx] if self.inputs[idx] else ""
        output_text = self.outputs[idx]
        
        # Format as instruction-following example
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + output_text
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (only compute loss on response part)
        labels = tokenized['input_ids'].clone()
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        labels[:, :prompt_tokens.size(1)] = -100  # Ignore prompt in loss
        
        return {
            'input_ids': tokenized['input_ids'].flatten(),
            'attention_mask': tokenized['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

# Example instruction tuning data
def create_instruction_dataset():
    """Create sample instruction tuning dataset"""
    
    instructions = [
        "Translate the following English text to French.",
        "Summarize the given paragraph in one sentence.",
        "Classify the sentiment of the following text.",
        "Generate a creative story beginning with the given sentence.",
        "Explain the concept in simple terms."
    ]
    
    inputs = [
        "Hello, how are you?",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        "I absolutely love this new restaurant!",
        "The old lighthouse stood alone on the rocky cliff.",
        "Quantum computing"
    ]
    
    outputs = [
        "Bonjour, comment allez-vous?",
        "Machine learning allows computers to learn from data without explicit programming.",
        "Positive",
        "The old lighthouse stood alone on the rocky cliff, its beacon cutting through the stormy night as Sarah approached with a mysterious package.",
        "Quantum computing uses the principles of quantum mechanics to process information in ways that classical computers cannot, potentially solving certain problems much faster."
    ]
    
    return instructions, inputs, outputs

def instruction_tune_model(model, tokenizer, use_lora=True):
    """Fine-tune model for instruction following"""
    
    # Apply LoRA if specified
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
    
    # Create dataset
    instructions, inputs, outputs = create_instruction_dataset()
    dataset = InstructionTuningDataset(instructions, inputs, outputs, tokenizer)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./instruction-tuned-model",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        fp16=True,
        remove_unused_columns=False,
    )
    
    # Custom trainer for causal LM
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x])
        }
    )
    
    trainer.train()
    return model
```

### RLHF (Reinforcement Learning from Human Feedback)

```python
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class RLHFTrainer:
    """Simplified RLHF trainer using PPO"""
    
    def __init__(self, policy_model, value_model, reward_model, tokenizer):
        self.policy_model = policy_model  # The model being trained
        self.value_model = value_model    # Value function estimator
        self.reward_model = reward_model  # Human preference model
        self.tokenizer = tokenizer
        
        self.kl_coeff = 0.1  # KL divergence coefficient
        self.clip_ratio = 0.2  # PPO clipping ratio

    def compute_rewards(self, queries, responses):
        """Compute rewards using the reward model"""
        rewards = []
        
        for query, response in zip(queries, responses):
            # Combine query and response
            full_text = query + response
            inputs = self.tokenizer(full_text, return_tensors='pt')
            
            with torch.no_grad():
                reward = self.reward_model(**inputs).logits.squeeze()
            
            rewards.append(reward)
        
        return torch.stack(rewards)

    def compute_advantages(self, rewards, values):
        """Compute advantages using Generalized Advantage Estimation"""
        advantages = []
        last_advantage = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + 0.99 * next_value - values[i]
            advantage = delta + 0.99 * 0.95 * last_advantage
            advantages.insert(0, advantage)
            last_advantage = advantage
        
        return torch.tensor(advantages)

    def ppo_step(self, queries, responses, old_log_probs, advantages):
        """Perform PPO update step"""
        
        # Compute new log probabilities
        new_log_probs = []
        for query, response in zip(queries, responses):
            inputs = self.tokenizer(query + response, return_tensors='pt')
            outputs = self.policy_model(**inputs)
            
            # Calculate log probabilities for the response tokens
            response_tokens = self.tokenizer(response, return_tensors='pt')['input_ids']
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            
            # Gather log probs for actual tokens
            selected_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
            new_log_probs.append(selected_log_probs.sum())
        
        new_log_probs = torch.stack(new_log_probs)
        
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute PPO loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add KL divergence penalty
        kl_div = (old_log_probs - new_log_probs).mean()
        total_loss = policy_loss + self.kl_coeff * kl_div
        
        return total_loss

    def train_step(self, batch_queries, batch_responses):
        """Single RLHF training step"""
        
        # Generate responses and compute initial log probs
        with torch.no_grad():
            old_log_probs = []
            for query, response in zip(batch_queries, batch_responses):
                inputs = self.tokenizer(query + response, return_tensors='pt')
                outputs = self.policy_model(**inputs)
                
                response_tokens = self.tokenizer(response, return_tensors='pt')['input_ids']
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                selected_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
                old_log_probs.append(selected_log_probs.sum())
            
            old_log_probs = torch.stack(old_log_probs)
        
        # Compute rewards
        rewards = self.compute_rewards(batch_queries, batch_responses)
        
        # Compute values
        values = []
        for query, response in zip(batch_queries, batch_responses):
            inputs = self.tokenizer(query + response, return_tensors='pt')
            with torch.no_grad():
                value = self.value_model(**inputs).logits.squeeze()
            values.append(value)
        values = torch.stack(values)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, values)
        
        # PPO update
        loss = self.ppo_step(batch_queries, batch_responses, old_log_probs, advantages)
        
        return loss
```

## 5. Fine-tuning for Specific Applications

### Domain-Specific Fine-tuning Examples

**1. Medical Text Analysis:**
```python
def finetune_medical_model():
    """Fine-tune for medical text analysis"""
    
    # Medical-specific LoRA configuration
    medical_lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32,  # Higher rank for complex domain
        lora_alpha=64,
        lora_dropout=0.05,  # Lower dropout for medical precision
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        bias="lora_only",
    )
    
    # Medical dataset preprocessing
    def preprocess_medical_text(text):
        # Handle medical abbreviations, terminology
        # Normalize drug names, dosages, etc.
        return text.lower().strip()
    
    # Custom loss function for medical tasks
    class MedicalClassificationLoss(nn.Module):
        def __init__(self, class_weights=None):
            super().__init__()
            self.class_weights = class_weights
            
        def forward(self, logits, labels):
            # Weighted cross-entropy for imbalanced medical data
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            return loss
    
    return medical_lora_config

# Example medical fine-tuning
medical_texts = [
    "Patient presents with acute chest pain and shortness of breath.",
    "Recommended dosage: 10mg twice daily with meals.",
    "No adverse reactions reported during clinical trial.",
]
medical_labels = [1, 0, 0]  # 1: urgent, 0: routine
```

**2. Legal Document Processing:**
```python
def finetune_legal_model():
    """Fine-tune for legal document analysis"""
    
    class LegalDocumentDataset:
        def __init__(self, documents, labels, tokenizer):
            self.documents = documents
            self.labels = labels
            self.tokenizer = tokenizer
            
        def preprocess_legal_text(self, text):
            # Handle legal citations, case references
            # Normalize section numbering
            return text
            
        def __getitem__(self, idx):
            doc = self.preprocess_legal_text(self.documents[idx])
            label = self.labels[idx]
            
            # Longer context for legal documents
            encoding = self.tokenizer(
                doc,
                truncation=True,
                padding='max_length',
                max_length=1024,  # Longer for legal texts
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label)
            }
    
    # Legal-specific configuration
    legal_config = {
        'learning_rate': 2e-5,  # Lower LR for stability
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
    }
    
    return legal_config
```

**3. Code Generation Fine-tuning:**
```python
def finetune_code_generation():
    """Fine-tune for code generation tasks"""
    
    class CodeGenerationDataset:
        def __init__(self, prompts, code_solutions, tokenizer):
            self.prompts = prompts
            self.code_solutions = code_solutions
            self.tokenizer = tokenizer
            
        def format_code_prompt(self, prompt, solution):
            return f"""# Task: {prompt}
# Language: Python

```python
{solution}
```"""
            
        def __getitem__(self, idx):
            prompt = self.prompts[idx]
            solution = self.code_solutions[idx]
            
            formatted_text = self.format_code_prompt(prompt, solution)
            
            encoding = self.tokenizer(
                formatted_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()
            }
    
    # Code-specific metrics
    def compute_code_metrics(predictions, references):
        """Compute code-specific evaluation metrics"""
        
        # BLEU score for code similarity
        from evaluate import load
        bleu = load("bleu")
        
        # Execution accuracy (if possible)
        execution_success = 0
        for pred, ref in zip(predictions, references):
            try:
                exec(pred)  # Try to execute generated code
                execution_success += 1
            except:
                pass
        
        execution_rate = execution_success / len(predictions)
        bleu_score = bleu.compute(predictions=predictions, references=references)
        
        return {
            'bleu': bleu_score['bleu'],
            'execution_rate': execution_rate
        }
    
    return CodeGenerationDataset, compute_code_metrics
```

## 6. Evaluation and Best Practices

### Comprehensive Evaluation Framework

```python
class FineTuningEvaluator:
    """Comprehensive evaluation for fine-tuned models"""
    
    def __init__(self, model, tokenizer, test_datasets):
        self.model = model
        self.tokenizer = tokenizer
        self.test_datasets = test_datasets
    
    def evaluate_performance(self, dataset_name, metric_types=['accuracy', 'f1']):
        """Evaluate model performance on test dataset"""
        
        test_dataset = self.test_datasets[dataset_name]
        predictions = []
        true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataset:
                inputs = {k: v.to(self.model.device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**inputs)
                
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].numpy())
        
        # Compute metrics
        results = {}
        if 'accuracy' in metric_types:
            from sklearn.metrics import accuracy_score
            results['accuracy'] = accuracy_score(true_labels, predictions)
        
        if 'f1' in metric_types:
            from sklearn.metrics import f1_score
            results['f1'] = f1_score(true_labels, predictions, average='weighted')
        
        return results
    
    def evaluate_efficiency(self):
        """Evaluate computational efficiency"""
        
        # Memory usage
        import psutil
        import time
        
        # Inference speed
        sample_input = "This is a test input for speed evaluation."
        inputs = self.tokenizer(sample_input, return_tensors='pt')
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(**inputs)
        
        # Timing
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = self.model(**inputs)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 100
        
        # Memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'avg_inference_time': avg_inference_time,
            'memory_usage_mb': memory_usage
        }
    
    def compare_with_baseline(self, baseline_model):
        """Compare fine-tuned model with baseline"""
        
        comparison_results = {}
        
        for dataset_name in self.test_datasets:
            # Fine-tuned model performance
            ft_results = self.evaluate_performance(dataset_name)
            
            # Baseline model performance
            self.model = baseline_model
            baseline_results = self.evaluate_performance(dataset_name)
            self.model = self.model  # Switch back
            
            # Compute improvements
            improvements = {}
            for metric, value in ft_results.items():
                baseline_value = baseline_results[metric]
                improvement = ((value - baseline_value) / baseline_value) * 100
                improvements[f'{metric}_improvement'] = improvement
            
            comparison_results[dataset_name] = {
                'fine_tuned': ft_results,
                'baseline': baseline_results,
                'improvements': improvements
            }
        
        return comparison_results

# Best practices checklist
def fine_tuning_checklist():
    """Best practices for fine-tuning transformer models"""
    
    best_practices = {
        'data_preparation': [
            "Ensure high-quality, diverse training data",
            "Handle class imbalance appropriately",
            "Validate data format and preprocessing",
            "Split data properly (train/val/test)"
        ],
        'model_selection': [
            "Choose appropriate base model for domain",
            "Consider model size vs. computational constraints",
            "Select parameter-efficient method based on resources",
            "Evaluate different rank values for LoRA"
        ],
        'training_configuration': [
            "Use appropriate learning rate scheduling",
            "Monitor for overfitting with validation loss",
            "Apply gradient clipping for stability",
            "Use mixed precision training when possible"
        ],
        'evaluation': [
            "Use domain-appropriate evaluation metrics",
            "Test on held-out data not seen during training",
            "Evaluate both performance and efficiency",
            "Compare with relevant baselines"
        ],
        'deployment': [
            "Test model behavior on edge cases",
            "Monitor for distribution shift in production",
            "Implement proper error handling",
            "Plan for regular model updates"
        ]
    }
    
    for category, practices in best_practices.items():
        print(f"\n{category.upper()}:")
        for practice in practices:
            print(f"  ✓ {practice}")
    
    return best_practices
```

## 7. Learning Objectives

By the end of this section, you should be able to:
- **Implement** various parameter-efficient fine-tuning methods (LoRA, adapters, prompt tuning)
- **Compare** trade-offs between different fine-tuning approaches
- **Apply** instruction tuning for improving model following capabilities
- **Understand** RLHF principles and implementation
- **Fine-tune** models for domain-specific applications
- **Evaluate** fine-tuned models comprehensively
- **Follow** best practices for stable and effective fine-tuning

### Self-Assessment Checklist

□ Can implement LoRA from scratch and using PEFT library  
□ Can explain the theory behind different parameter-efficient methods  
□ Can create instruction-tuning datasets and training pipelines  
□ Can fine-tune models for specific domains (medical, legal, code)  
□ Can evaluate fine-tuned models using appropriate metrics  
□ Can debug common fine-tuning issues (overfitting, instability)  
□ Can optimize fine-tuning for computational efficiency  

## 8. Practical Exercises

**Exercise 1: LoRA Implementation**
```python
# TODO: Implement LoRA from scratch for a small transformer
# Compare with PEFT library implementation
# Measure parameter reduction and performance retention
```

**Exercise 2: Domain-Specific Fine-tuning**
```python
# TODO: Choose a domain (medical, legal, scientific)
# Collect or simulate domain-specific data
# Fine-tune using different methods and compare results
```

**Exercise 3: Instruction Tuning Pipeline**
```python
# TODO: Create instruction-tuning dataset
# Implement training pipeline with proper evaluation
# Test model's ability to follow diverse instructions
```

## 9. Study Materials

### Essential Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

### Tools and Libraries
```bash
pip install peft transformers datasets
pip install accelerate bitsandbytes
pip install evaluate scikit-learn
pip install wandb  # For experiment tracking
```

### Online Resources
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Fine-tuning Tutorial](https://huggingface.co/blog/lora)
- [Parameter-Efficient Fine-tuning Guide](https://github.com/huggingface/peft)
