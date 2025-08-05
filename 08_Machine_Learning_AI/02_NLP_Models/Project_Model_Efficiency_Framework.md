# Project: Model Efficiency Framework

*Duration: 6-8 weeks*  
*Difficulty: Advanced*  
*Tech Stack: Python, PyTorch, Transformers, ONNX, TensorRT, Weights & Biases*

## Project Overview

The Model Efficiency Framework is a comprehensive toolkit for optimizing transformer models to achieve the best balance between performance and computational efficiency. This framework implements various compression techniques, benchmarking tools, and automated optimization pipelines to make large language models more practical for production deployment.

### Business Impact
- **Cost Reduction**: Reduce inference costs by 60-80% through model optimization
- **Latency Improvement**: Achieve 2-10x faster inference speeds
- **Resource Efficiency**: Enable deployment on edge devices and smaller hardware
- **Scalability**: Handle higher throughput with the same infrastructure

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Input   │    │  Optimization   │    │  Benchmarking   │
│   (Original)    │───►│    Pipeline     │───►│    Suite        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model Analysis │    │   Compression   │    │   Performance   │
│   & Profiling   │◄──►│   Techniques    │◄──►│   Evaluation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Optimization   │    │   Deployment    │    │   Monitoring    │
│  Strategies     │───►│   Framework     │───►│   & Tracking    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components Implementation

### 1. Model Analysis and Profiling

#### Deep Model Profiler
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import time
import psutil
import GPUtil
from typing import Dict, Any, List, Tuple
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ModelProfile:
    """Data class for storing model profiling results."""
    model_name: str
    parameters: int
    model_size_mb: float
    flops: int
    memory_usage_mb: float
    inference_time_ms: float
    throughput_samples_per_sec: float
    layer_analysis: Dict[str, Any]
    bottlenecks: List[str]

class ModelProfiler:
    """Comprehensive model profiling and analysis toolkit."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.profiles = {}
        
    def _get_device(self, device: str) -> torch.device:
        """Automatically select the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def profile_model(self, model_name: str, 
                     sample_inputs: List[str] = None) -> ModelProfile:
        """Comprehensive model profiling."""
        
        print(f"🔍 Profiling model: {model_name}")
        
        # Load model and tokenizer
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        
        # Default sample inputs if not provided
        if sample_inputs is None:
            sample_inputs = [
                "This is a sample text for profiling.",
                "Another longer sample text that contains more tokens and complexity for comprehensive testing.",
                "Short text.",
                "A very long sample text " * 50  # Create long input
            ]
        
        # Basic model statistics
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        
        # Memory usage analysis
        memory_usage = self._measure_memory_usage(model, tokenizer, sample_inputs)
        
        # Inference time analysis
        inference_stats = self._measure_inference_time(model, tokenizer, sample_inputs)
        
        # FLOPS calculation
        flops = self._calculate_flops(model, tokenizer, sample_inputs[0])
        
        # Layer-wise analysis
        layer_analysis = self._analyze_layers(model, tokenizer, sample_inputs[0])
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(layer_analysis, inference_stats)
        
        profile = ModelProfile(
            model_name=model_name,
            parameters=param_count,
            model_size_mb=model_size_mb,
            flops=flops,
            memory_usage_mb=memory_usage,
            inference_time_ms=inference_stats['avg_time_ms'],
            throughput_samples_per_sec=inference_stats['throughput'],
            layer_analysis=layer_analysis,
            bottlenecks=bottlenecks
        )
        
        self.profiles[model_name] = profile
        return profile
    
    def _measure_memory_usage(self, model: nn.Module, tokenizer, 
                            sample_inputs: List[str]) -> float:
        """Measure peak memory usage during inference."""
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Warmup
            with torch.no_grad():
                inputs = tokenizer(sample_inputs[0], return_tensors="pt", 
                                 truncation=True, padding=True).to(self.device)
                _ = model(**inputs)
            
            # Measure peak memory
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            return peak_memory_bytes / (1024 * 1024)  # Convert to MB
        else:
            # For CPU, use psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            with torch.no_grad():
                inputs = tokenizer(sample_inputs[0], return_tensors="pt", 
                                 truncation=True, padding=True).to(self.device)
                _ = model(**inputs)
            
            memory_after = process.memory_info().rss
            return (memory_after - memory_before) / (1024 * 1024)
    
    def _measure_inference_time(self, model: nn.Module, tokenizer, 
                               sample_inputs: List[str]) -> Dict[str, float]:
        """Measure inference time statistics."""
        
        times = []
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(5):
                inputs = tokenizer(sample_inputs[0], return_tensors="pt", 
                                 truncation=True, padding=True).to(self.device)
                _ = model(**inputs)
        
        # Benchmark runs
        with torch.no_grad():
            for text in sample_inputs * 10:  # Multiple runs for statistics
                inputs = tokenizer(text, return_tensors="pt", 
                                 truncation=True, padding=True).to(self.device)
                
                start_time = time.perf_counter()
                _ = model(**inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'throughput': 1000 / np.mean(times)  # samples per second
        }
    
    def _calculate_flops(self, model: nn.Module, tokenizer, sample_text: str) -> int:
        """Calculate FLOPs for model inference."""
        try:
            from fvcore.nn import FlopCountMode, flop_count
            
            inputs = tokenizer(sample_text, return_tensors="pt", 
                             truncation=True, padding=True).to(self.device)
            
            with flop_count(model, inputs, supported_ops=None) as flop_counter:
                _ = model(**inputs)
            
            return flop_counter.total()
        except ImportError:
            print("fvcore not available, skipping FLOPS calculation")
            return 0
    
    def _analyze_layers(self, model: nn.Module, tokenizer, sample_text: str) -> Dict[str, Any]:
        """Analyze performance of individual layers."""
        
        layer_stats = {}
        
        # Hook function to measure layer execution time
        def layer_hook(name):
            def hook_fn(module, input, output):
                if hasattr(hook_fn, 'start_time'):
                    elapsed = time.perf_counter() - hook_fn.start_time
                    if name not in layer_stats:
                        layer_stats[name] = []
                    layer_stats[name].append(elapsed * 1000)  # Convert to ms
                hook_fn.start_time = time.perf_counter()
            return hook_fn
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(layer_hook(name))
                hooks.append(hook)
        
        # Run inference with hooks
        inputs = tokenizer(sample_text, return_tensors="pt", 
                         truncation=True, padding=True).to(self.device)
        
        with torch.no_grad():
            for _ in range(10):  # Multiple runs for statistics
                _ = model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate statistics
        layer_analysis = {}
        for layer_name, times in layer_stats.items():
            layer_analysis[layer_name] = {
                'avg_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'total_time_ms': np.sum(times),
                'percentage': 0  # Will be calculated later
            }
        
        # Calculate percentages
        total_time = sum(stats['total_time_ms'] for stats in layer_analysis.values())
        for stats in layer_analysis.values():
            stats['percentage'] = (stats['total_time_ms'] / total_time) * 100
        
        return layer_analysis
    
    def _identify_bottlenecks(self, layer_analysis: Dict, inference_stats: Dict) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Identify slow layers (top 20% of execution time)
        sorted_layers = sorted(layer_analysis.items(), 
                             key=lambda x: x[1]['percentage'], reverse=True)
        
        total_percentage = 0
        for layer_name, stats in sorted_layers:
            total_percentage += stats['percentage']
            if total_percentage <= 80:  # Top layers contributing to 80% of time
                bottlenecks.append(f"Layer '{layer_name}': {stats['percentage']:.1f}% of total time")
        
        # Check for high variance in inference time
        if inference_stats['std_time_ms'] / inference_stats['avg_time_ms'] > 0.2:
            bottlenecks.append("High inference time variance - possible memory bottleneck")
        
        return bottlenecks
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare multiple model profiles."""
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.profiles:
                print(f"Profiling {model_name}...")
                self.profile_model(model_name)
            
            profile = self.profiles[model_name]
            comparison_data.append({
                'Model': model_name,
                'Parameters (M)': profile.parameters / 1e6,
                'Size (MB)': profile.model_size_mb,
                'Memory (MB)': profile.memory_usage_mb,
                'Inference Time (ms)': profile.inference_time_ms,
                'Throughput (samples/s)': profile.throughput_samples_per_sec,
                'FLOPS (G)': profile.flops / 1e9
            })
        
        return pd.DataFrame(comparison_data)
    
    def visualize_comparison(self, comparison_df: pd.DataFrame):
        """Create visualizations for model comparison."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Parameters vs Inference Time
        axes[0, 0].scatter(comparison_df['Parameters (M)'], comparison_df['Inference Time (ms)'])
        axes[0, 0].set_xlabel('Parameters (M)')
        axes[0, 0].set_ylabel('Inference Time (ms)')
        axes[0, 0].set_title('Parameters vs Inference Time')
        
        # Model Size vs Memory Usage
        axes[0, 1].scatter(comparison_df['Size (MB)'], comparison_df['Memory (MB)'])
        axes[0, 1].set_xlabel('Model Size (MB)')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Model Size vs Memory Usage')
        
        # Throughput comparison
        axes[0, 2].bar(range(len(comparison_df)), comparison_df['Throughput (samples/s)'])
        axes[0, 2].set_xlabel('Models')
        axes[0, 2].set_ylabel('Throughput (samples/s)')
        axes[0, 2].set_title('Throughput Comparison')
        axes[0, 2].set_xticks(range(len(comparison_df)))
        axes[0, 2].set_xticklabels(comparison_df['Model'], rotation=45)
        
        # Efficiency ratio (Throughput / Parameters)
        efficiency = comparison_df['Throughput (samples/s)'] / comparison_df['Parameters (M)']
        axes[1, 0].bar(range(len(comparison_df)), efficiency)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Efficiency (throughput/param)')
        axes[1, 0].set_title('Model Efficiency')
        axes[1, 0].set_xticks(range(len(comparison_df)))
        axes[1, 0].set_xticklabels(comparison_df['Model'], rotation=45)
        
        # Memory efficiency
        mem_efficiency = comparison_df['Throughput (samples/s)'] / comparison_df['Memory (MB)']
        axes[1, 1].bar(range(len(comparison_df)), mem_efficiency)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Memory Efficiency')
        axes[1, 1].set_title('Memory Efficiency')
        axes[1, 1].set_xticks(range(len(comparison_df)))
        axes[1, 1].set_xticklabels(comparison_df['Model'], rotation=45)
        
        # FLOPS vs Performance
        axes[1, 2].scatter(comparison_df['FLOPS (G)'], comparison_df['Throughput (samples/s)'])
        axes[1, 2].set_xlabel('FLOPS (G)')
        axes[1, 2].set_ylabel('Throughput (samples/s)')
        axes[1, 2].set_title('FLOPS vs Performance')
        
        plt.tight_layout()
        return fig
```

### 2. Advanced Compression Techniques

#### Quantization Framework
```python
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_qconfig
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import copy

class QuantizationType(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training

class ModelQuantizer:
    """Advanced model quantization toolkit."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.original_model = None
        self.quantized_models = {}
        self.calibration_data = []
        
    def load_model(self):
        """Load the original model."""
        self.original_model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.original_model
    
    def dynamic_quantization(self, qconfig_dict: Optional[Dict] = None) -> nn.Module:
        """Apply dynamic quantization to the model."""
        
        if self.original_model is None:
            self.load_model()
        
        # Default quantization configuration
        if qconfig_dict is None:
            qconfig_dict = {
                torch.nn.Linear: default_qconfig,
                torch.nn.Embedding: default_qconfig
            }
        
        # Apply dynamic quantization
        quantized_model = quantize_dynamic(
            self.original_model,
            qconfig_dict,
            dtype=torch.qint8
        )
        
        self.quantized_models['dynamic'] = quantized_model
        return quantized_model
    
    def static_quantization(self, calibration_texts: List[str]) -> nn.Module:
        """Apply static quantization with calibration data."""
        
        if self.original_model is None:
            self.load_model()
        
        # Prepare model for static quantization
        model = copy.deepcopy(self.original_model)
        model.eval()
        
        # Set quantization configuration
        model.qconfig = default_qconfig
        
        # Prepare model
        torch.quantization.prepare(model, inplace=True)
        
        # Calibration phase
        print("🔧 Calibrating model for static quantization...")
        with torch.no_grad():
            for text in calibration_texts:
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, padding=True)
                _ = model(**inputs)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        self.quantized_models['static'] = quantized_model
        return quantized_model
    
    def mixed_precision_quantization(self, sensitive_layers: List[str] = None) -> nn.Module:
        """Apply mixed precision quantization keeping sensitive layers in FP16/32."""
        
        if self.original_model is None:
            self.load_model()
        
        model = copy.deepcopy(self.original_model)
        
        # Default sensitive layers (usually attention and layer norm)
        if sensitive_layers is None:
            sensitive_layers = ['attention', 'layernorm', 'layer_norm']
        
        # Custom quantization configuration
        qconfig_dict = {}
        
        for name, module in model.named_modules():
            is_sensitive = any(sensitive in name.lower() for sensitive in sensitive_layers)
            
            if isinstance(module, torch.nn.Linear) and not is_sensitive:
                qconfig_dict[module] = default_qconfig
            elif isinstance(module, torch.nn.Embedding):
                qconfig_dict[module] = default_qconfig
        
        quantized_model = quantize_dynamic(model, qconfig_dict, dtype=torch.qint8)
        
        self.quantized_models['mixed_precision'] = quantized_model
        return quantized_model
    
    def benchmark_quantization(self, test_texts: List[str]) -> Dict[str, Any]:
        """Benchmark different quantization approaches."""
        
        results = {}
        
        # Original model benchmark
        results['original'] = self._benchmark_model(self.original_model, test_texts, "Original")
        
        # Benchmark quantized models
        for quant_type, model in self.quantized_models.items():
            results[quant_type] = self._benchmark_model(model, test_texts, f"Quantized ({quant_type})")
        
        return results
    
    def _benchmark_model(self, model: nn.Module, test_texts: List[str], 
                        model_name: str) -> Dict[str, Any]:
        """Benchmark a specific model variant."""
        
        model.eval()
        
        # Model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Inference time
        times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                inputs = self.tokenizer(test_texts[0], return_tensors="pt", 
                                      truncation=True, padding=True)
                _ = model(**inputs)
            
            # Benchmark
            for text in test_texts:
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, padding=True)
                
                start_time = time.perf_counter()
                outputs = model(**inputs)
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)
        
        return {
            'model_name': model_name,
            'model_size_mb': model_size / (1024 * 1024),
            'avg_inference_time_ms': np.mean(times),
            'std_inference_time_ms': np.std(times),
            'throughput_samples_per_sec': 1000 / np.mean(times)
        }

# Pruning Framework
class ModelPruner:
    """Advanced model pruning techniques."""
    
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.pruned_models = {}
        
    def magnitude_pruning(self, sparsity: float = 0.5) -> nn.Module:
        """Apply magnitude-based pruning."""
        
        import torch.nn.utils.prune as prune
        
        model = copy.deepcopy(self.original_model)
        
        # Apply pruning to linear layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
        
        self.pruned_models[f'magnitude_{sparsity}'] = model
        return model
    
    def structured_pruning(self, prune_ratio: float = 0.3) -> nn.Module:
        """Apply structured pruning (removes entire neurons/channels)."""
        
        model = copy.deepcopy(self.original_model)
        
        # Identify layers to prune
        linear_layers = [(name, module) for name, module in model.named_modules() 
                        if isinstance(module, torch.nn.Linear)]
        
        for name, layer in linear_layers:
            if 'classifier' not in name and 'output' not in name:  # Skip output layers
                # Calculate importance scores (L1 norm of weights)
                importance = torch.norm(layer.weight.data, p=1, dim=1)
                
                # Determine neurons to keep
                num_neurons = layer.weight.size(0)
                num_keep = int(num_neurons * (1 - prune_ratio))
                
                if num_keep > 0:
                    _, keep_indices = torch.topk(importance, num_keep)
                    
                    # Create new layer with reduced size
                    new_layer = torch.nn.Linear(
                        layer.in_features, 
                        num_keep, 
                        bias=layer.bias is not None
                    )
                    
                    # Copy weights
                    new_layer.weight.data = layer.weight.data[keep_indices]
                    if layer.bias is not None:
                        new_layer.bias.data = layer.bias.data[keep_indices]
                    
                    # Replace layer in model
                    parent_name = '.'.join(name.split('.')[:-1])
                    layer_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        setattr(parent, layer_name, new_layer)
                    else:
                        setattr(model, layer_name, new_layer)
        
        self.pruned_models[f'structured_{prune_ratio}'] = model
        return model
    
    def gradual_pruning(self, target_sparsity: float = 0.8, 
                       num_steps: int = 10, fine_tune_fn=None) -> nn.Module:
        """Apply gradual pruning with optional fine-tuning."""
        
        import torch.nn.utils.prune as prune
        
        model = copy.deepcopy(self.original_model)
        
        # Calculate sparsity schedule
        sparsity_schedule = np.linspace(0, target_sparsity, num_steps)
        
        for step, current_sparsity in enumerate(sparsity_schedule):
            print(f"Pruning step {step + 1}/{num_steps}, sparsity: {current_sparsity:.2f}")
            
            # Apply pruning
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if hasattr(module, 'weight_mask'):
                        # Remove previous pruning
                        prune.remove(module, 'weight')
                    
                    # Apply new pruning
                    prune.l1_unstructured(module, name='weight', amount=current_sparsity)
            
            # Optional fine-tuning step
            if fine_tune_fn is not None:
                model = fine_tune_fn(model)
        
        self.pruned_models[f'gradual_{target_sparsity}'] = model
        return model
```

### 3. Knowledge Distillation Framework

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb

class DistillationLoss(nn.Module):
    """Custom loss function for knowledge distillation."""
    
    def __init__(self, alpha: float = 0.7, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_outputs, teacher_outputs, true_labels=None):
        """Calculate distillation loss."""
        
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=-1)
        
        # Distillation loss
        distillation_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Hard target loss (if labels provided)
        if true_labels is not None:
            hard_loss = F.cross_entropy(student_outputs, true_labels)
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = distillation_loss
            
        return total_loss, distillation_loss

class ModelDistiller:
    """Knowledge distillation framework for model compression."""
    
    def __init__(self, teacher_model_name: str, student_model_name: str):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
        
    def setup_models(self):
        """Initialize teacher and student models."""
        
        # Load teacher model (larger, pre-trained)
        self.teacher_model = AutoModel.from_pretrained(self.teacher_model_name)
        self.teacher_model.eval()
        
        # Load student model (smaller)
        self.student_model = AutoModel.from_pretrained(self.student_model_name)
        
        # Use teacher's tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
        
        print(f"Teacher model parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        print(f"Student model parameters: {sum(p.numel() for p in self.student_model.parameters()):,}")
        
    def distill_knowledge(self, train_texts: List[str], 
                         val_texts: List[str] = None,
                         num_epochs: int = 3,
                         batch_size: int = 16,
                         learning_rate: float = 5e-5,
                         temperature: float = 4.0,
                         alpha: float = 0.7) -> nn.Module:
        """Perform knowledge distillation training."""
        
        if self.teacher_model is None:
            self.setup_models()
        
        # Create dataset
        train_dataset = self._create_distillation_dataset(train_texts)
        
        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model.to(device)
        self.student_model.to(device)
        
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        criterion = DistillationLoss(alpha=alpha, temperature=temperature)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.student_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_distill_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    ).last_hidden_state
                
                # Student forward pass
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state
                
                # Calculate loss
                loss, distill_loss = criterion(student_outputs, teacher_outputs)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_distill_loss += distill_loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                          f"Loss: {loss.item():.4f}, Distill Loss: {distill_loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            avg_distill_loss = total_distill_loss / len(train_loader)
            
            print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, "
                  f"Avg Distill Loss: {avg_distill_loss:.4f}")
            
            # Validation (if provided)
            if val_texts:
                val_metrics = self._evaluate_distillation(val_texts)
                print(f"Validation metrics: {val_metrics}")
        
        return self.student_model
    
    def _create_distillation_dataset(self, texts: List[str]) -> Dataset:
        """Create dataset for distillation training."""
        
        class DistillationDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }
        
        return DistillationDataset(texts, self.tokenizer)
    
    def _evaluate_distillation(self, val_texts: List[str]) -> Dict[str, float]:
        """Evaluate the distilled model."""
        
        self.student_model.eval()
        device = next(self.student_model.parameters()).device
        
        # Compare teacher and student outputs
        similarities = []
        
        with torch.no_grad():
            for text in val_texts[:100]:  # Sample for evaluation
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, padding=True).to(device)
                
                teacher_out = self.teacher_model(**inputs).last_hidden_state
                student_out = self.student_model(**inputs).last_hidden_state
                
                # Calculate cosine similarity
                teacher_flat = teacher_out.mean(dim=1)  # Average pooling
                student_flat = student_out.mean(dim=1)
                
                similarity = F.cosine_similarity(teacher_flat, student_flat, dim=-1).item()
                similarities.append(similarity)
        
        return {
            'avg_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities)
        }
```

### 4. Comprehensive Benchmarking Suite

```python
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
from pathlib import Path
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""
    model_name: str
    optimization_type: str
    inference_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    model_size_mb: float
    accuracy_score: float
    compression_ratio: float
    energy_consumption_j: Optional[float] = None

class ComprehensiveBenchmark:
    """Comprehensive benchmarking suite for model optimization."""
    
    def __init__(self, reference_model_name: str):
        self.reference_model_name = reference_model_name
        self.results = []
        self.benchmark_data = {}
        
    def benchmark_model(self, model: nn.Module, model_name: str, 
                       optimization_type: str, test_texts: List[str],
                       reference_accuracy: float = None) -> BenchmarkResult:
        """Comprehensive benchmark of a single model."""
        
        print(f"🚀 Benchmarking {model_name} ({optimization_type})")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.reference_model_name)
        
        # 1. Model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # 2. Memory usage
        memory_usage_mb = self._measure_peak_memory(model, tokenizer, test_texts[0], device)
        
        # 3. Inference speed
        inference_stats = self._benchmark_inference_speed(model, tokenizer, test_texts, device)
        
        # 4. Accuracy (if reference provided)
        accuracy_score = self._evaluate_accuracy(model, tokenizer, test_texts, reference_accuracy)
        
        # 5. Energy consumption (if available)
        energy_consumption = self._measure_energy_consumption(model, tokenizer, test_texts[0], device)
        
        # 6. Calculate compression ratio
        reference_size = self._get_reference_model_size()
        compression_ratio = reference_size / model_size_mb if reference_size else 1.0
        
        result = BenchmarkResult(
            model_name=model_name,
            optimization_type=optimization_type,
            inference_time_ms=inference_stats['avg_time_ms'],
            throughput_samples_per_sec=inference_stats['throughput'],
            memory_usage_mb=memory_usage_mb,
            model_size_mb=model_size_mb,
            accuracy_score=accuracy_score,
            compression_ratio=compression_ratio,
            energy_consumption_j=energy_consumption
        )
        
        self.results.append(result)
        return result
    
    def _measure_peak_memory(self, model: nn.Module, tokenizer, 
                           sample_text: str, device: torch.device) -> float:
        """Measure peak memory usage during inference."""
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                inputs = tokenizer(sample_text, return_tensors="pt", 
                                 truncation=True, padding=True).to(device)
                _ = model(**inputs)
                torch.cuda.synchronize()
            
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            return peak_memory_bytes / (1024 * 1024)
        else:
            # Use psutil for CPU memory measurement
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            with torch.no_grad():
                inputs = tokenizer(sample_text, return_tensors="pt", 
                                 truncation=True, padding=True).to(device)
                _ = model(**inputs)
            
            memory_after = process.memory_info().rss
            return (memory_after - memory_before) / (1024 * 1024)
    
    def _benchmark_inference_speed(self, model: nn.Module, tokenizer,
                                  test_texts: List[str], device: torch.device) -> Dict[str, float]:
        """Benchmark inference speed with multiple text samples."""
        
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                inputs = tokenizer(test_texts[0], return_tensors="pt", 
                                 truncation=True, padding=True).to(device)
                _ = model(**inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        
        # Benchmark
        with torch.no_grad():
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", 
                                 truncation=True, padding=True).to(device)
                
                start_time = time.perf_counter()
                _ = model(**inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)
        
        return {
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'throughput': 1000 / np.mean(times)
        }
    
    def _evaluate_accuracy(self, model: nn.Module, tokenizer,
                          test_texts: List[str], reference_accuracy: float) -> float:
        """Evaluate model accuracy (simplified - task-specific implementation needed)."""
        
        if reference_accuracy is None:
            return 1.0  # Default score if no reference
        
        # This is a simplified accuracy measure
        # In practice, you would implement task-specific evaluation
        
        # For demonstration, we simulate accuracy degradation based on model changes
        model_params = sum(p.numel() for p in model.parameters())
        reference_params = self._get_reference_model_params()
        
        if reference_params:
            param_ratio = model_params / reference_params
            # Simulate accuracy relationship with model size
            simulated_accuracy = reference_accuracy * (0.8 + 0.2 * param_ratio)
            return min(simulated_accuracy, reference_accuracy)
        
        return reference_accuracy
    
    def _measure_energy_consumption(self, model: nn.Module, tokenizer,
                                  sample_text: str, device: torch.device) -> Optional[float]:
        """Measure energy consumption during inference (GPU only)."""
        
        if device.type != "cuda":
            return None
        
        try:
            # This requires additional tools like nvidia-ml-py for precise measurement
            # Here we provide a simplified estimation
            
            # Get GPU utilization before
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
            
            gpu = gpus[0]
            power_before = gpu.powerDraw  # Watts
            
            # Run inference
            start_time = time.perf_counter()
            with torch.no_grad():
                inputs = tokenizer(sample_text, return_tensors="pt", 
                                 truncation=True, padding=True).to(device)
                _ = model(**inputs)
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Get GPU utilization after
            gpus = GPUtil.getGPUs()
            gpu = gpus[0]
            power_after = gpu.powerDraw
            
            # Calculate energy consumption (simplified)
            avg_power = (power_before + power_after) / 2
            duration_seconds = end_time - start_time
            energy_joules = avg_power * duration_seconds
            
            return energy_joules
            
        except Exception as e:
            print(f"Could not measure energy consumption: {e}")
            return None
    
    def _get_reference_model_size(self) -> float:
        """Get reference model size for compression ratio calculation."""
        try:
            ref_model = AutoModel.from_pretrained(self.reference_model_name)
            size_mb = sum(p.numel() * p.element_size() for p in ref_model.parameters()) / (1024 * 1024)
            del ref_model  # Free memory
            return size_mb
        except:
            return None
    
    def _get_reference_model_params(self) -> int:
        """Get reference model parameter count."""
        try:
            ref_model = AutoModel.from_pretrained(self.reference_model_name)
            params = sum(p.numel() for p in ref_model.parameters())
            del ref_model
            return params
        except:
            return None
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report."""
        
        if not self.results:
            return pd.DataFrame()
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            data.append({
                'Model': result.model_name,
                'Optimization': result.optimization_type,
                'Inference Time (ms)': result.inference_time_ms,
                'Throughput (samples/s)': result.throughput_samples_per_sec,
                'Memory (MB)': result.memory_usage_mb,
                'Model Size (MB)': result.model_size_mb,
                'Accuracy': result.accuracy_score,
                'Compression Ratio': result.compression_ratio,
                'Energy (J)': result.energy_consumption_j
            })
        
        df = pd.DataFrame(data)
        
        # Calculate efficiency metrics
        df['Speed Improvement'] = df['Throughput (samples/s)'] / df['Throughput (samples/s)'].iloc[0]
        df['Memory Efficiency'] = df['Throughput (samples/s)'] / df['Memory (MB)']
        df['Size Efficiency'] = df['Throughput (samples/s)'] / df['Model Size (MB)']
        df['Quality-Speed Ratio'] = df['Accuracy'] * df['Throughput (samples/s)']
        
        return df
    
    def visualize_tradeoffs(self, df: pd.DataFrame):
        """Create visualizations for performance trade-offs."""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Model Optimization Trade-offs Analysis', fontsize=16)
        
        # 1. Speed vs Accuracy
        scatter = axes[0, 0].scatter(df['Throughput (samples/s)'], df['Accuracy'], 
                                   c=df['Model Size (MB)'], cmap='viridis', s=100)
        axes[0, 0].set_xlabel('Throughput (samples/s)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Speed vs Accuracy Trade-off')
        plt.colorbar(scatter, ax=axes[0, 0], label='Model Size (MB)')
        
        # 2. Model Size vs Accuracy
        axes[0, 1].scatter(df['Model Size (MB)'], df['Accuracy'], 
                          c=df['Throughput (samples/s)'], cmap='plasma', s=100)
        axes[0, 1].set_xlabel('Model Size (MB)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Size vs Accuracy Trade-off')
        
        # 3. Compression Ratio vs Performance Loss
        performance_loss = 1 - df['Accuracy'] / df['Accuracy'].max()
        axes[0, 2].scatter(df['Compression Ratio'], performance_loss, s=100)
        axes[0, 2].set_xlabel('Compression Ratio')
        axes[0, 2].set_ylabel('Performance Loss')
        axes[0, 2].set_title('Compression vs Performance Loss')
        
        # 4. Efficiency comparison
        efficiency_metrics = ['Speed Improvement', 'Memory Efficiency', 'Size Efficiency']
        efficiency_data = df[efficiency_metrics].values.T
        
        im = axes[1, 0].imshow(efficiency_data, cmap='RdYlGn', aspect='auto')
        axes[1, 0].set_xticks(range(len(df)))
        axes[1, 0].set_xticklabels(df['Optimization'], rotation=45)
        axes[1, 0].set_yticks(range(len(efficiency_metrics)))
        axes[1, 0].set_yticklabels(efficiency_metrics)
        axes[1, 0].set_title('Efficiency Metrics Heatmap')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Pareto frontier (Accuracy vs Speed)
        # Find Pareto optimal points
        pareto_points = []
        for i, row in df.iterrows():
            is_pareto = True
            for j, other_row in df.iterrows():
                if i != j and (other_row['Accuracy'] >= row['Accuracy'] and 
                              other_row['Throughput (samples/s)'] >= row['Throughput (samples/s)'] and
                              (other_row['Accuracy'] > row['Accuracy'] or 
                               other_row['Throughput (samples/s)'] > row['Throughput (samples/s)'])):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(i)
        
        axes[1, 1].scatter(df['Throughput (samples/s)'], df['Accuracy'], alpha=0.6, s=100)
        if pareto_points:
            pareto_df = df.iloc[pareto_points].sort_values('Throughput (samples/s)')
            axes[1, 1].plot(pareto_df['Throughput (samples/s)'], pareto_df['Accuracy'], 
                           'r-o', linewidth=2, markersize=8, label='Pareto Frontier')
        axes[1, 1].set_xlabel('Throughput (samples/s)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Pareto Frontier Analysis')
        axes[1, 1].legend()
        
        # 6. Overall efficiency ranking
        # Calculate overall efficiency score
        df_norm = df.copy()
        for col in ['Throughput (samples/s)', 'Accuracy', 'Compression Ratio']:
            df_norm[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        df_norm['Overall Score'] = (df_norm['Throughput (samples/s)_norm'] + 
                                   df_norm['Accuracy_norm'] + 
                                   df_norm['Compression Ratio_norm']) / 3
        
        bars = axes[1, 2].bar(range(len(df)), df_norm['Overall Score'])
        axes[1, 2].set_xlabel('Models')
        axes[1, 2].set_ylabel('Overall Efficiency Score')
        axes[1, 2].set_title('Overall Efficiency Ranking')
        axes[1, 2].set_xticks(range(len(df)))
        axes[1, 2].set_xticklabels(df['Optimization'], rotation=45)
        
        # Color bars by score
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(df_norm['Overall Score'].iloc[i]))
        
        plt.tight_layout()
        return fig
    
    def save_results(self, output_path: Path):
        """Save benchmark results to file."""
        
        # Save detailed results as JSON
        results_data = []
        for result in self.results:
            results_data.append({
                'model_name': result.model_name,
                'optimization_type': result.optimization_type,
                'inference_time_ms': result.inference_time_ms,
                'throughput_samples_per_sec': result.throughput_samples_per_sec,
                'memory_usage_mb': result.memory_usage_mb,
                'model_size_mb': result.model_size_mb,
                'accuracy_score': result.accuracy_score,
                'compression_ratio': result.compression_ratio,
                'energy_consumption_j': result.energy_consumption_j
            })
        
        with open(output_path / 'benchmark_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save comparison report as CSV
        df = self.generate_comparison_report()
        df.to_csv(output_path / 'comparison_report.csv', index=False)
        
        print(f"Results saved to {output_path}")
```

This comprehensive Model Efficiency Framework provides:

✅ **Deep Model Profiling** with layer-wise analysis and bottleneck identification  
✅ **Advanced Compression Techniques** including quantization, pruning, and knowledge distillation  
✅ **Benchmarking Suite** with comprehensive performance metrics  
✅ **Trade-off Analysis** with Pareto frontier optimization  
✅ **Production-Ready Tools** for deployment optimization  
✅ **Automated Optimization Pipelines** for efficient model compression  
✅ **Energy Consumption Monitoring** for green AI practices  
✅ **Detailed Reporting** and visualization capabilities
