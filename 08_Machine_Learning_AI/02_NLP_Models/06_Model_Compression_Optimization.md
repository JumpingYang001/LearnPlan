# Model Compression and Optimization

*Duration: 2 weeks*

## Overview

Model compression and optimization are essential for deploying large language models in resource-constrained environments. These techniques reduce model size, memory usage, and inference time while maintaining acceptable performance levels.

## 1. Quantization Techniques

### Understanding Quantization

Quantization reduces the precision of model weights and activations from 32-bit floating point to lower precision formats (16-bit, 8-bit, or even 4-bit).

**Precision Comparison:**
```
FP32: ████████████████████████████████ (32 bits)
FP16: ████████████████                 (16 bits) 
INT8: ████████                         (8 bits)
INT4: ████                             (4 bits)
```

### Post-Training Quantization (PTQ)

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class SimpleQuantizer:
    """Simple quantization implementation for educational purposes"""
    
    @staticmethod
    def quantize_tensor(tensor, num_bits=8, signed=True):
        """Quantize a tensor to specified bit precision"""
        
        if signed:
            qmin, qmax = -(2**(num_bits-1)), 2**(num_bits-1) - 1
        else:
            qmin, qmax = 0, 2**num_bits - 1
        
        # Find min and max values
        min_val, max_val = tensor.min().item(), tensor.max().item()
        
        # Calculate scale and zero point
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = max(qmin, min(qmax, round(zero_point)))
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        return quantized, scale, zero_point
    
    @staticmethod
    def dequantize_tensor(quantized_tensor, scale, zero_point):
        """Dequantize tensor back to float"""
        return scale * (quantized_tensor - zero_point)

def demonstrate_quantization():
    """Demonstrate quantization effects"""
    
    # Original tensor
    original = torch.randn(100, 100) * 10
    
    # Quantize to different precisions
    precisions = [8, 4, 2]
    
    print(f"Original tensor stats:")
    print(f"  Mean: {original.mean():.4f}, Std: {original.std():.4f}")
    print(f"  Min: {original.min():.4f}, Max: {original.max():.4f}")
    print(f"  Size: {original.numel() * 4} bytes (FP32)")
    print()
    
    for bits in precisions:
        quantized, scale, zero_point = SimpleQuantizer.quantize_tensor(original, bits)
        dequantized = SimpleQuantizer.dequantize_tensor(quantized, scale, zero_point)
        
        # Calculate error
        error = torch.mean((original - dequantized) ** 2).sqrt()
        compression_ratio = 32 / bits
        
        print(f"INT{bits} Quantization:")
        print(f"  RMSE: {error:.4f}")
        print(f"  Compression: {compression_ratio:.1f}x")
        print(f"  Size: {original.numel() * bits // 8} bytes")
        print()

# Run demonstration
demonstrate_quantization()
```

### Dynamic Quantization with PyTorch

```python
import torch.quantization as quant

def apply_dynamic_quantization(model):
    """Apply dynamic quantization to a model"""
    
    # Prepare model for quantization
    model.eval()
    
    # Apply dynamic quantization
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},  # Target layers
        dtype=torch.qint8
    )
    
    return quantized_model

def compare_model_sizes(original_model, quantized_model):
    """Compare model sizes and inference speed"""
    import time
    
    def get_model_size(model):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024 / 1024
    
    # Size comparison
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    # Speed comparison (requires actual inference)
    sample_input = torch.randn(1, 512)  # Example input
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original_model(sample_input)
            _ = quantized_model(sample_input)
    
    # Time original model
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = original_model(sample_input)
    original_time = time.time() - start_time
    
    # Time quantized model
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = quantized_model(sample_input)
    quantized_time = time.time() - start_time
    
    print(f"Original inference time: {original_time:.4f}s")
    print(f"Quantized inference time: {quantized_time:.4f}s")
    print(f"Speedup: {original_time / quantized_time:.2f}x")

# Example usage
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# quantized_model = apply_dynamic_quantization(model)
# compare_model_sizes(model, quantized_model)
```

### Advanced Quantization with BitsAndBytes

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_model_with_quantization(model_name, quantization_type="8bit"):
    """Load model with BitsAndBytes quantization"""
    
    if quantization_type == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=True
        )
    elif quantization_type == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        quantization_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return model

def benchmark_quantization_methods():
    """Benchmark different quantization methods"""
    
    model_name = "microsoft/DialoGPT-small"  # Small model for testing
    methods = ["fp16", "8bit", "4bit"]
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} quantization...")
        
        try:
            if method == "fp16":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                model = load_model_with_quantization(model_name, method)
            
            # Measure memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
                
                # Simple forward pass
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                inputs = tokenizer("Hello, how are you?", return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
                
                results[method] = {
                    'memory_mb': memory_used,
                    'success': True
                }
            else:
                results[method] = {'success': True, 'memory_mb': 'N/A (CPU)'}
                
        except Exception as e:
            print(f"Error with {method}: {str(e)}")
            results[method] = {'success': False, 'error': str(e)}
    
    return results

# Run benchmark
# results = benchmark_quantization_methods()
# for method, result in results.items():
#     print(f"{method}: {result}")
```

## 2. Knowledge Distillation

### Teacher-Student Framework

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining task loss and distillation loss"""
    
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha  # Balance between task loss and distillation loss
        self.temperature = temperature  # Softmax temperature for distillation
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute distillation loss
        
        Args:
            student_logits: Predictions from student model
            teacher_logits: Predictions from teacher model  
            labels: Ground truth labels
        """
        
        # Task loss (student vs ground truth)
        task_loss = F.cross_entropy(student_logits, labels)
        
        # Distillation loss (student vs teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss
        
        return total_loss, task_loss, distill_loss

class TransformerDistillation:
    """Framework for distilling transformer models"""
    
    def __init__(self, teacher_model, student_model, tokenizer):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
    def distill_step(self, batch, criterion):
        """Single distillation training step"""
        
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']
        
        # Teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Student predictions
        student_outputs = self.student_model(**inputs)
        student_logits = student_outputs.logits
        
        # Compute distillation loss
        loss, task_loss, distill_loss = criterion(
            student_logits, teacher_logits, labels
        )
        
        return loss, {
            'total_loss': loss.item(),
            'task_loss': task_loss.item(),
            'distill_loss': distill_loss.item()
        }
    
    def train_student(self, train_dataloader, num_epochs=3, learning_rate=5e-5):
        """Train student model using knowledge distillation"""
        
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(), 
            lr=learning_rate
        )
        criterion = DistillationLoss(alpha=0.7, temperature=4.0)
        
        self.student_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_task_loss = 0
            total_distill_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                loss, metrics = self.distill_step(batch, criterion)
                loss.backward()
                optimizer.step()
                
                total_loss += metrics['total_loss']
                total_task_loss += metrics['task_loss']
                total_distill_loss += metrics['distill_loss']
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}")
                    print(f"  Total Loss: {metrics['total_loss']:.4f}")
                    print(f"  Task Loss: {metrics['task_loss']:.4f}")
                    print(f"  Distill Loss: {metrics['distill_loss']:.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            avg_task_loss = total_task_loss / len(train_dataloader)
            avg_distill_loss = total_distill_loss / len(train_dataloader)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Avg Total Loss: {avg_loss:.4f}")
            print(f"  Avg Task Loss: {avg_task_loss:.4f}")
            print(f"  Avg Distill Loss: {avg_distill_loss:.4f}")

# Example: Distill BERT-large to BERT-base
def distill_bert_example():
    """Example of distilling BERT-large to BERT-base"""
    
    # Load models
    teacher_name = "bert-large-uncased"
    student_name = "bert-base-uncased"
    
    teacher_model = AutoModel.from_pretrained(teacher_name)
    student_model = AutoModel.from_pretrained(student_name)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    
    # Add classification heads
    teacher_classifier = nn.Linear(teacher_model.config.hidden_size, 2)
    student_classifier = nn.Linear(student_model.config.hidden_size, 2)
    
    # Combine model + classifier
    class ClassificationModel(nn.Module):
        def __init__(self, base_model, classifier):
            super().__init__()
            self.base_model = base_model
            self.classifier = classifier
            
        def forward(self, **inputs):
            outputs = self.base_model(**inputs)
            pooled_output = outputs.pooler_output
            logits = self.classifier(pooled_output)
            return type('Outputs', (), {'logits': logits})()
    
    teacher_full = ClassificationModel(teacher_model, teacher_classifier)
    student_full = ClassificationModel(student_model, student_classifier)
    
    # Create distillation framework
    distiller = TransformerDistillation(teacher_full, student_full, tokenizer)
    
    return distiller

# distiller = distill_bert_example()
```

### Attention Transfer and Hidden State Matching

```python
class AdvancedDistillationLoss(nn.Module):
    """Advanced distillation with attention and hidden state matching"""
    
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha  # Task loss weight
        self.beta = beta    # Attention loss weight
        self.gamma = gamma  # Hidden state loss weight
        self.temperature = temperature
        
    def attention_loss(self, student_attentions, teacher_attentions):
        """Compute attention transfer loss"""
        att_loss = 0
        
        # Match number of layers (teacher might have more layers)
        student_layers = len(student_attentions)
        teacher_layers = len(teacher_attentions)
        
        layer_mapping = [i * teacher_layers // student_layers 
                        for i in range(student_layers)]
        
        for s_idx, t_idx in enumerate(layer_mapping):
            student_att = student_attentions[s_idx]
            teacher_att = teacher_attentions[t_idx]
            
            # Average over heads if necessary
            if student_att.dim() == 4:  # [batch, heads, seq, seq]
                student_att = student_att.mean(dim=1)
                teacher_att = teacher_att.mean(dim=1)
            
            # MSE loss between attention matrices
            att_loss += F.mse_loss(student_att, teacher_att)
        
        return att_loss / student_layers
    
    def hidden_state_loss(self, student_hidden, teacher_hidden):
        """Compute hidden state matching loss"""
        hidden_loss = 0
        
        student_layers = len(student_hidden)
        teacher_layers = len(teacher_hidden)
        
        layer_mapping = [i * teacher_layers // student_layers 
                        for i in range(student_layers)]
        
        for s_idx, t_idx in enumerate(layer_mapping):
            student_h = student_hidden[s_idx]
            teacher_h = teacher_hidden[t_idx]
            
            # Project to same dimension if needed
            if student_h.size(-1) != teacher_h.size(-1):
                projection = nn.Linear(
                    teacher_h.size(-1), 
                    student_h.size(-1)
                ).to(teacher_h.device)
                teacher_h = projection(teacher_h)
            
            hidden_loss += F.mse_loss(student_h, teacher_h)
        
        return hidden_loss / student_layers
    
    def forward(self, student_outputs, teacher_outputs, labels):
        """Compute comprehensive distillation loss"""
        
        # Task loss
        task_loss = F.cross_entropy(student_outputs.logits, labels)
        
        # Distillation loss (logits)
        student_soft = F.log_softmax(
            student_outputs.logits / self.temperature, dim=-1
        )
        teacher_soft = F.softmax(
            teacher_outputs.logits / self.temperature, dim=-1
        )
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distill_loss *= (self.temperature ** 2)
        
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss
        
        # Attention loss (if available)
        if hasattr(student_outputs, 'attentions') and hasattr(teacher_outputs, 'attentions'):
            if student_outputs.attentions and teacher_outputs.attentions:
                att_loss = self.attention_loss(
                    student_outputs.attentions, 
                    teacher_outputs.attentions
                )
                total_loss += self.beta * att_loss
        
        # Hidden state loss (if available)
        if hasattr(student_outputs, 'hidden_states') and hasattr(teacher_outputs, 'hidden_states'):
            if student_outputs.hidden_states and teacher_outputs.hidden_states:
                hidden_loss = self.hidden_state_loss(
                    student_outputs.hidden_states,
                    teacher_outputs.hidden_states
                )
                total_loss += self.gamma * hidden_loss
        
        return total_loss
```

## 3. Pruning and Sparse Models

### Structured vs Unstructured Pruning

```python
import torch
import torch.nn.utils.prune as prune
import numpy as np

class ModelPruner:
    """Comprehensive model pruning toolkit"""
    
    def __init__(self, model):
        self.model = model
        self.original_params = sum(p.numel() for p in model.parameters())
    
    def magnitude_based_pruning(self, pruning_ratio=0.2, structured=False):
        """Prune based on weight magnitude"""
        
        if structured:
            # Structured pruning: remove entire neurons/channels
            self._structured_magnitude_pruning(pruning_ratio)
        else:
            # Unstructured pruning: remove individual weights
            self._unstructured_magnitude_pruning(pruning_ratio)
    
    def _unstructured_magnitude_pruning(self, pruning_ratio):
        """Remove individual weights with smallest magnitude"""
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                
                # Optionally prune bias
                if hasattr(module, 'bias') and module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=pruning_ratio)
    
    def _structured_magnitude_pruning(self, pruning_ratio):
        """Remove entire neurons/channels"""
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune entire neurons (output channels)
                prune.ln_structured(
                    module, name='weight', amount=pruning_ratio, n=2, dim=0
                )
            elif isinstance(module, nn.Conv2d):
                # Prune entire filters (output channels)
                prune.ln_structured(
                    module, name='weight', amount=pruning_ratio, n=2, dim=0
                )
    
    def gradual_magnitude_pruning(self, initial_sparsity=0.0, final_sparsity=0.8, 
                                 pruning_frequency=100, pruning_steps=1000):
        """Gradually increase pruning during training"""
        
        def pruning_schedule(step):
            if step >= pruning_steps:
                return final_sparsity
            
            progress = step / pruning_steps
            sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * progress
            return sparsity
        
        return pruning_schedule
    
    def gradient_based_pruning(self, dataloader, num_samples=1000):
        """Prune based on gradient information (Fisher information)"""
        
        # Compute Fisher information
        fisher_info = {}
        
        self.model.eval()
        for name, param in self.model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients (Fisher information approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            
            sample_count += batch['labels'].size(0)
        
        # Normalize Fisher information
        for name in fisher_info:
            fisher_info[name] /= sample_count
        
        # Prune parameters with low Fisher information
        self._fisher_based_pruning(fisher_info, pruning_ratio=0.2)
    
    def _fisher_based_pruning(self, fisher_info, pruning_ratio):
        """Prune based on Fisher information"""
        
        # Collect all Fisher scores
        all_scores = []
        param_info = []
        
        for name, param in self.model.named_parameters():
            if name in fisher_info:
                scores = fisher_info[name].flatten()
                all_scores.append(scores)
                param_info.extend([(name, i) for i in range(len(scores))])
        
        all_scores = torch.cat(all_scores)
        
        # Find pruning threshold
        num_to_prune = int(len(all_scores) * pruning_ratio)
        threshold = torch.topk(all_scores, num_to_prune, largest=False)[0][-1]
        
        # Apply pruning
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if name + '.weight' in fisher_info:
                    mask = fisher_info[name + '.weight'] > threshold
                    prune.custom_from_mask(module, name='weight', mask=mask)
    
    def analyze_sparsity(self):
        """Analyze current model sparsity"""
        
        total_params = 0
        zero_params = 0
        
        sparsity_by_layer = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    # Pruned parameters
                    weight = module.weight_orig
                    mask = module.weight_mask
                    
                    layer_total = weight.numel()
                    layer_zeros = (mask == 0).sum().item()
                    
                    total_params += layer_total
                    zero_params += layer_zeros
                    
                    sparsity_by_layer[name] = layer_zeros / layer_total
                else:
                    # Unpruned parameters
                    weight = module.weight
                    layer_total = weight.numel()
                    layer_zeros = (weight == 0).sum().item()
                    
                    total_params += layer_total
                    zero_params += layer_zeros
                    
                    sparsity_by_layer[name] = layer_zeros / layer_total
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0
        
        print(f"Overall Sparsity: {overall_sparsity:.2%}")
        print(f"Parameters: {total_params - zero_params:,} / {total_params:,}")
        print(f"Compression Ratio: {total_params / (total_params - zero_params):.2f}x")
        print("\nPer-layer sparsity:")
        for layer, sparsity in sparsity_by_layer.items():
            print(f"  {layer}: {sparsity:.2%}")
        
        return overall_sparsity, sparsity_by_layer
    
    def remove_pruning_reparametrization(self):
        """Make pruning permanent by removing masks"""
        
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass  # Module wasn't pruned
                
                try:
                    prune.remove(module, 'bias')
                except:
                    pass  # Bias wasn't pruned

# Example usage
def demonstrate_pruning():
    """Demonstrate different pruning techniques"""
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 25)
            self.fc3 = nn.Linear(25, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleModel()
    pruner = ModelPruner(model)
    
    print("Original model:")
    original_sparsity, _ = pruner.analyze_sparsity()
    
    # Apply magnitude-based pruning
    pruner.magnitude_based_pruning(pruning_ratio=0.3)
    
    print("\nAfter magnitude-based pruning (30%):")
    pruned_sparsity, layer_sparsity = pruner.analyze_sparsity()
    
    return model, pruner

# model, pruner = demonstrate_pruning()
```

## 4. Optimized Transformer Models

### Model Architecture Optimizations

```python
class OptimizedTransformerBlock(nn.Module):
    """Optimized transformer block with various efficiency improvements"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, 
                 use_flash_attention=False, use_rotary_pe=True):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_flash_attention = use_flash_attention
        self.use_rotary_pe = use_rotary_pe
        
        # Multi-head attention with optimizations
        if use_flash_attention:
            self.attention = FlashMultiHeadAttention(d_model, n_heads, dropout)
        else:
            self.attention = OptimizedMultiHeadAttention(d_model, n_heads, dropout)
        
        # Optimized feed-forward network
        self.ffn = OptimizedFFN(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm for better training)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Rotary positional embeddings
        if use_rotary_pe:
            self.rotary_pe = RotaryPositionalEmbedding(d_model // n_heads)
    
    def forward(self, x, mask=None, cache=None):
        # Pre-norm + attention + residual
        normed = self.ln1(x)
        
        if self.use_rotary_pe:
            attn_out, new_cache = self.attention(normed, mask, cache, self.rotary_pe)
        else:
            attn_out, new_cache = self.attention(normed, mask, cache)
        
        x = x + attn_out
        
        # Pre-norm + FFN + residual
        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x, new_cache

class OptimizedMultiHeadAttention(nn.Module):
    """Memory and compute optimized multi-head attention"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Single linear layer for Q, K, V (more efficient)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.d_k)
    
    def forward(self, x, mask=None, cache=None, rotary_pe=None):
        batch_size, seq_len, d_model = x.shape
        
        # Single matrix multiplication for Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply rotary positional embeddings
        if rotary_pe is not None:
            cos, sin = rotary_pe(seq_len, x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Use KV cache if provided
        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Efficient attention computation
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention (if available)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0
            )
        else:
            # Manual attention computation
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.out_proj(attn_output)
        
        # Return new cache
        new_cache = (k, v) if cache is not None else None
        
        return output, new_cache

class OptimizedFFN(nn.Module):
    """Optimized feed-forward network with activation checkpointing"""
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swiglu':
            # SwiGLU activation (used in LLAMA)
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        if hasattr(self, 'gate_proj'):
            # SwiGLU activation
            gate = self.activation(self.gate_proj(x))
            up = self.linear1(x)
            x = gate * up
        else:
            # Standard activation
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.linear2(x)
        return x

# Flash Attention implementation (simplified)
class FlashMultiHeadAttention(nn.Module):
    """Simplified Flash Attention implementation"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # Similar to OptimizedMultiHeadAttention but with tiling optimizations
        # In practice, would use flash-attn library
        self.attention = OptimizedMultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, x, mask=None, cache=None, rotary_pe=None):
        # Flash attention optimizes memory by computing attention in tiles
        # This is a placeholder - real implementation would use CUDA kernels
        return self.attention(x, mask, cache, rotary_pe)
```

### Memory Optimization Techniques

```python
class MemoryOptimizedTraining:
    """Memory optimization techniques for training large models"""
    
    def __init__(self, model):
        self.model = model
    
    def apply_gradient_checkpointing(self):
        """Apply gradient checkpointing to save memory"""
        from torch.utils.checkpoint import checkpoint
        
        # Replace forward methods with checkpointed versions
        def create_checkpointed_forward(original_forward):
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(original_forward, *args, **kwargs)
            return checkpointed_forward
        
        for module in self.model.modules():
            if isinstance(module, OptimizedTransformerBlock):
                module.forward = create_checkpointed_forward(module.forward)
    
    def apply_activation_checkpointing(self, segments=2):
        """Divide model into segments and checkpoint between them"""
        
        # Group layers into segments
        layers = [m for m in self.model.modules() 
                 if isinstance(m, OptimizedTransformerBlock)]
        
        segment_size = len(layers) // segments
        
        for i in range(0, len(layers), segment_size):
            segment = layers[i:i+segment_size]
            
            # Wrap each segment with checkpointing
            def create_segment_forward(segment_layers):
                def segment_forward(x):
                    for layer in segment_layers:
                        x = layer(x)
                    return x
                return segment_forward
            
            segment_forward = create_segment_forward(segment)
            
            # Replace with checkpointed version
            def checkpointed_segment(x):
                return torch.utils.checkpoint.checkpoint(segment_forward, x)
    
    def optimize_memory_layout(self):
        """Optimize memory layout for better cache performance"""
        
        # Convert model to channels-last memory format (for conv layers)
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                module = module.to(memory_format=torch.channels_last)
        
        # Fuse operations where possible
        torch.jit.optimize_for_inference(self.model)
    
    def estimate_memory_usage(self, batch_size, seq_len):
        """Estimate memory usage for given input size"""
        
        # Rough estimation formulas
        d_model = self.model.config.hidden_size if hasattr(self.model, 'config') else 768
        n_layers = len([m for m in self.model.modules() 
                       if isinstance(m, OptimizedTransformerBlock)])
        
        # Parameter memory (weights)
        param_memory = sum(p.numel() * p.element_size() 
                          for p in self.model.parameters()) / 1024**3  # GB
        
        # Activation memory (rough estimate)
        activation_memory = (
            batch_size * seq_len * d_model * n_layers * 4  # 4 bytes per float32
        ) / 1024**3  # GB
        
        # Gradient memory (same as parameters for fp32)
        gradient_memory = param_memory
        
        # Optimizer states (Adam: 2x parameters)
        optimizer_memory = param_memory * 2
        
        total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory
        
        print(f"Memory Estimation for batch_size={batch_size}, seq_len={seq_len}:")
        print(f"  Parameters: {param_memory:.2f} GB")
        print(f"  Activations: {activation_memory:.2f} GB")
        print(f"  Gradients: {gradient_memory:.2f} GB")
        print(f"  Optimizer: {optimizer_memory:.2f} GB")
        print(f"  Total: {total_memory:.2f} GB")
        
        return total_memory

# Example usage
def optimize_model_for_deployment():
    """Complete optimization pipeline for model deployment"""
    
    # Load model (example)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("=== Original Model ===")
    original_size = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {original_size:,}")
    
    # 1. Apply quantization
    print("\n=== Applying Quantization ===")
    quantized_model = apply_dynamic_quantization(model)
    
    # 2. Apply pruning
    print("\n=== Applying Pruning ===")
    pruner = ModelPruner(quantized_model)
    pruner.magnitude_based_pruning(pruning_ratio=0.3)
    sparsity, _ = pruner.analyze_sparsity()
    
    # 3. Knowledge distillation (if we had a larger teacher model)
    print("\n=== Knowledge Distillation ===")
    print("Would apply distillation from larger teacher model...")
    
    # 4. Optimize for inference
    print("\n=== Inference Optimization ===")
    memory_optimizer = MemoryOptimizedTraining(quantized_model)
    total_memory = memory_optimizer.estimate_memory_usage(1, 512)
    
    # 5. Final model statistics
    final_size = sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)
    compression_ratio = original_size / final_size
    
    print(f"\n=== Final Results ===")
    print(f"Original parameters: {original_size:,}")
    print(f"Final parameters: {final_size:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Sparsity: {sparsity:.2%}")
    
    return quantized_model, tokenizer

# model, tokenizer = optimize_model_for_deployment()
```

## 5. Learning Objectives

By the end of this section, you should be able to:
- **Apply** various quantization techniques (INT8, INT4, dynamic, static)
- **Implement** knowledge distillation for model compression
- **Use** pruning methods to reduce model size and computational requirements
- **Optimize** transformer architectures for efficient inference
- **Evaluate** trade-offs between compression and performance
- **Deploy** compressed models in resource-constrained environments

### Self-Assessment Checklist

□ Can implement post-training quantization from scratch  
□ Can set up knowledge distillation training pipeline  
□ Can apply structured and unstructured pruning  
□ Can analyze model sparsity and compression ratios  
□ Can optimize memory usage during training and inference  
□ Can benchmark compressed models for speed and accuracy  
□ Can choose appropriate compression techniques for different deployment scenarios  

## 6. Practical Exercises

**Exercise 1: Quantization Comparison**
```python
# TODO: Compare different quantization methods on a transformer model
# Measure accuracy retention, speed improvement, and memory reduction
# Test on multiple tasks (classification, generation)
```

**Exercise 2: Custom Distillation**
```python
# TODO: Implement knowledge distillation from BERT-large to BERT-base
# Include attention transfer and hidden state matching
# Evaluate on GLUE benchmark tasks
```

**Exercise 3: Optimal Compression Pipeline**
```python
# TODO: Design optimal compression pipeline for specific deployment target
# Combine quantization, pruning, and distillation
# Achieve target latency/memory constraints while maximizing accuracy
```

## 7. Study Materials

### Essential Papers
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
- [BERT-of-Theseus: Compressing BERT by Progressive Module Replacing](https://arxiv.org/abs/2002.02925)

### Tools and Libraries
```bash
pip install torch torchvision
pip install transformers accelerate
pip install bitsandbytes optimum
pip install torch-pruning  # For advanced pruning
```

### Benchmarking Tools
- Model size and speed profiling
- Memory usage monitoring
- Accuracy evaluation frameworks
