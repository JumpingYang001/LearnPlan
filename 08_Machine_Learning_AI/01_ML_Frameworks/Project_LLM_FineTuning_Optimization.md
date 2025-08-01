# Project: LLM Fine-tuning and Optimization

*Duration: 4-6 weeks*  
*Difficulty: Advanced*  
*Prerequisites: Python, PyTorch, Machine Learning fundamentals*

## Project Overview

This comprehensive project guides you through the complete pipeline of Large Language Model (LLM) fine-tuning, optimization, and deployment. You'll learn to adapt pre-trained models for specific tasks, optimize them for production environments, and deploy them with efficient inference pipelines.

### What You'll Build
- **Custom fine-tuned LLM** for domain-specific tasks (e.g., code generation, customer support, creative writing)
- **Optimized inference engine** with quantization and pruning techniques
- **Production-ready deployment pipeline** with API endpoints and monitoring
- **Performance benchmarking system** for comparing optimization techniques

## Learning Objectives

By completing this project, you will:
- ✅ **Master LLM fine-tuning** using HuggingFace Transformers and PyTorch
- ✅ **Implement optimization techniques** including quantization, pruning, and distillation
- ✅ **Build production deployment pipelines** with FastAPI and containerization
- ✅ **Apply performance optimization** for memory and speed improvements
- ✅ **Understand distributed training** and gradient accumulation strategies
- ✅ **Implement monitoring and evaluation** systems for model performance

## Key Features & Technologies

### Core Technologies
- **🤗 HuggingFace Transformers**: Model loading, fine-tuning, and tokenization
- **🔥 PyTorch**: Deep learning framework and CUDA optimization
- **⚡ TensorRT/ONNX**: Inference optimization and acceleration
- **🚀 FastAPI**: RESTful API development for model serving
- **🐳 Docker**: Containerization and deployment
- **📊 Weights & Biases**: Experiment tracking and monitoring

### Optimization Techniques
- **Quantization**: INT8/FP16 precision reduction
- **Pruning**: Structured and unstructured weight removal
- **Knowledge Distillation**: Teacher-student model compression
- **LoRA/QLoRA**: Parameter-efficient fine-tuning
- **Flash Attention**: Memory-efficient attention mechanisms

## Phase 1: Environment Setup and Data Preparation

### 1.1 Development Environment Setup

**Requirements Installation:**
```bash
# Create virtual environment
python -m venv llm_project
source llm_project/bin/activate  # On Windows: llm_project\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate evaluate
pip install bitsandbytes peft wandb
pip install fastapi uvicorn gradio streamlit
pip install tensorrt onnx onnxruntime-gpu
pip install scikit-learn matplotlib seaborn
```

**Project Structure Setup:**
```bash
mkdir llm_finetuning_project
cd llm_finetuning_project

# Create directory structure
mkdir -p {data,models,configs,scripts,notebooks,deployment,logs}
mkdir -p data/{raw,processed,validation}
mkdir -p models/{base,finetuned,optimized}
mkdir -p deployment/{api,docker,kubernetes}
```

### 1.2 Data Preparation Pipeline

**Dataset Loading and Preprocessing:**
```python
# scripts/data_preparation.py
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import json
import logging

class DataPreprocessor:
    def __init__(self, model_name="microsoft/DialoGPT-medium", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_conversational_data(self, conversations):
        """
        Prepare conversational data for fine-tuning
        Format: [{"input": "user message", "output": "assistant response"}]
        """
        formatted_data = []
        
        for conv in conversations:
            # Format conversation for training
            input_text = f"User: {conv['input']}\nAssistant:"
            target_text = f" {conv['output']}{self.tokenizer.eos_token}"
            
            formatted_data.append({
                "input_text": input_text,
                "target_text": target_text,
                "full_text": input_text + target_text
            })
        
        return formatted_data
    
    def tokenize_function(self, examples):
        """Tokenize the dataset for training"""
        # Tokenize inputs and targets
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            examples["target_text"],
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def create_dataset(self, data_path, test_size=0.2, val_size=0.1):
        """Create train/validation/test datasets"""
        # Load your custom data
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Prepare data
        formatted_data = self.prepare_conversational_data(raw_data)
        
        # Split data
        train_data, temp_data = train_test_split(
            formatted_data, test_size=test_size + val_size, random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=test_size/(test_size + val_size), random_state=42
        )
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
        test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

# Example usage
if __name__ == "__main__":
    # Sample data creation (replace with your actual data)
    sample_conversations = [
        {"input": "What is machine learning?", 
         "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
        {"input": "Explain neural networks", 
         "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information."},
        # Add more conversations...
    ]
    
    # Save sample data
    with open("data/raw/conversations.json", "w") as f:
        json.dump(sample_conversations, f, indent=2)
    
    # Prepare datasets
    preprocessor = DataPreprocessor()
    datasets = preprocessor.create_dataset("data/raw/conversations.json")
    
    print(f"Train dataset size: {len(datasets['train'])}")
    print(f"Validation dataset size: {len(datasets['validation'])}")
    print(f"Test dataset size: {len(datasets['test'])}")
```

**Alternative: Using Public Datasets:**
```python
# scripts/load_public_dataset.py
from datasets import load_dataset

def load_alpaca_dataset():
    """Load and prepare Alpaca instruction dataset"""
    dataset = load_dataset("tatsu-lab/alpaca")
    
    def format_alpaca(example):
        if example["input"]:
            instruction = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        else:
            instruction = f"### Instruction:\n{example['instruction']}\n\n### Response:"
        
        return {
            "input_text": instruction,
            "target_text": f"{example['output']}</s>",
            "full_text": instruction + f"{example['output']}</s>"
        }
    
    return dataset.map(format_alpaca)

def load_dolly_dataset():
    """Load and prepare Dolly instruction dataset"""
    dataset = load_dataset("databricks/databricks-dolly-15k")
    
    def format_dolly(example):
        context = f"\n\nContext: {example['context']}" if example['context'] else ""
        instruction = f"### Instruction:\n{example['instruction']}{context}\n\n### Response:"
        
        return {
            "input_text": instruction,
            "target_text": f"{example['response']}</s>",
            "full_text": instruction + f"{example['response']}</s>"
        }
    
    return dataset.map(format_dolly)
```

## Phase 2: Model Fine-tuning Implementation

### 2.1 Base Model Selection and Loading

**Model Selection Strategy:**
```python
# scripts/model_selection.py
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GPT2LMHeadModel,
    LlamaForCausalLM
)
import torch

class ModelSelector:
    """Helper class for selecting and loading appropriate models"""
    
    MODEL_CONFIGS = {
        "small": {
            "model_name": "microsoft/DialoGPT-small",
            "params": "117M",
            "memory_gb": 1,
            "use_case": "Testing and development"
        },
        "medium": {
            "model_name": "microsoft/DialoGPT-medium", 
            "params": "345M",
            "memory_gb": 2,
            "use_case": "Balanced performance"
        },
        "large": {
            "model_name": "microsoft/DialoGPT-large",
            "params": "762M", 
            "memory_gb": 4,
            "use_case": "High quality responses"
        },
        "llama2-7b": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "params": "7B",
            "memory_gb": 16,
            "use_case": "Production quality"
        }
    }
    
    @staticmethod
    def get_available_models():
        """Display available model configurations"""
        for size, config in ModelSelector.MODEL_CONFIGS.items():
            print(f"{size.upper()}:")
            print(f"  Model: {config['model_name']}")
            print(f"  Parameters: {config['params']}")
            print(f"  Memory: ~{config['memory_gb']}GB")
            print(f"  Use case: {config['use_case']}\n")
    
    @staticmethod
    def load_model_and_tokenizer(model_size="medium", device="auto"):
        """Load model and tokenizer based on size selection"""
        config = ModelSelector.MODEL_CONFIGS[model_size]
        model_name = config["model_name"]
        
        print(f"Loading {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        return model, tokenizer

# Example usage
if __name__ == "__main__":
    ModelSelector.get_available_models()
    model, tokenizer = ModelSelector.load_model_and_tokenizer("medium")
### 2.2 Full Fine-tuning Implementation

**Complete Training Pipeline:**
```python
# scripts/full_finetuning.py
import torch
import torch.nn as nn
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
import wandb
import os
from datetime import datetime
import json

class LLMFineTuner:
    def __init__(self, model, tokenizer, output_dir="./models/finetuned"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        
        # Initialize Weights & Biases for tracking
        wandb.init(
            project="llm-finetuning",
            name=f"finetuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    
    def setup_training_args(self, 
                          num_epochs=3,
                          batch_size=4,
                          learning_rate=5e-5,
                          warmup_steps=500,
                          gradient_accumulation_steps=4,
                          fp16=True):
        """Configure training arguments"""
        
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            eval_steps=500,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            dataloader_num_workers=4,
            remove_unused_columns=False,
            label_names=["labels"],
            prediction_loss_only=True,
        )
    
    def create_data_collator(self):
        """Create data collator for language modeling"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8 if self.tokenizer.pad_token_id is not None else None,
        )
    
    def train(self, train_dataset, eval_dataset, training_args):
        """Execute the training process"""
        
        # Create data collator
        data_collator = self.create_data_collator()
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        print("Starting training...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        trainer.save_state()
        
        # Log training results
        wandb.log({
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_steps_per_second": train_result.metrics["train_steps_per_second"]
        })
        
        print(f"Training completed!")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        
        return trainer, train_result
    
    def evaluate_model(self, trainer, test_dataset):
        """Evaluate the fine-tuned model"""
        print("Evaluating model...")
        
        eval_results = trainer.evaluate(test_dataset)
        
        # Log evaluation results
        wandb.log({
            "eval_loss": eval_results["eval_loss"],
            "eval_runtime": eval_results["eval_runtime"],
            "eval_steps_per_second": eval_results["eval_steps_per_second"]
        })
        
        print(f"Evaluation loss: {eval_results['eval_loss']:.4f}")
        return eval_results

# Training script example
if __name__ == "__main__":
    from model_selection import ModelSelector
    from data_preparation import DataPreprocessor
    
    # Load model and tokenizer
    model, tokenizer = ModelSelector.load_model_and_tokenizer("medium")
    
    # Prepare datasets
    preprocessor = DataPreprocessor(model_name="microsoft/DialoGPT-medium")
    datasets = preprocessor.create_dataset("data/raw/conversations.json")
    
    # Initialize fine-tuner
    fine_tuner = LLMFineTuner(model, tokenizer)
    
    # Setup training arguments
    training_args = fine_tuner.setup_training_args(
        num_epochs=5,
        batch_size=2,  # Adjust based on GPU memory
        learning_rate=3e-5,
        gradient_accumulation_steps=8
    )
    
    # Train the model
    trainer, results = fine_tuner.train(
        datasets["train"], 
        datasets["validation"], 
        training_args
    )
    
    # Evaluate the model
    eval_results = fine_tuner.evaluate_model(trainer, datasets["test"])
```

### 2.3 Parameter-Efficient Fine-tuning with LoRA

**LoRA Implementation for Memory Efficiency:**
```python
# scripts/lora_finetuning.py
import torch
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

class LoRAFineTuner:
    def __init__(self, base_model_name, use_4bit=True):
        self.base_model_name = base_model_name
        self.use_4bit = use_4bit
        
    def setup_quantization_config(self):
        """Configure 4-bit quantization for memory efficiency"""
        if self.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None
    
    def load_model_for_lora(self):
        """Load model with quantization for LoRA training"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        quantization_config = self.setup_quantization_config()
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        return model, tokenizer
    
    def setup_lora_config(self, 
                         r=16, 
                         lora_alpha=32, 
                         lora_dropout=0.05,
                         target_modules=None):
        """Configure LoRA parameters"""
        
        if target_modules is None:
            # Common target modules for different architectures
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    
    def prepare_model_for_training(self, model, lora_config):
        """Prepare model with LoRA adapters"""
        
        # Prepare model for k-bit training if using quantization
        if self.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
    
    def train_with_lora(self, train_dataset, eval_dataset, output_dir="./models/lora"):
        """Complete LoRA training pipeline"""
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_for_lora()
        
        # Setup LoRA configuration
        lora_config = self.setup_lora_config()
        
        # Prepare model with LoRA
        model = self.prepare_model_for_training(model, lora_config)
        
        # Setup training arguments (more aggressive for LoRA)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,  # Can use larger batch size
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            learning_rate=2e-4,  # Higher learning rate for LoRA
            fp16=True,
            logging_steps=25,
            eval_steps=250,
            save_steps=250,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to="wandb",
            remove_unused_columns=False,
        )
        
        # Create trainer
        from transformers import DataCollatorForLanguageModeling
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save LoRA adapters
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return model, tokenizer, trainer

# Example usage
if __name__ == "__main__":
    # Initialize LoRA fine-tuner
    lora_tuner = LoRAFineTuner("microsoft/DialoGPT-medium", use_4bit=True)
    
    # Prepare your datasets (using previous data preparation code)
    # datasets = load_your_datasets()
    
    # Train with LoRA
    model, tokenizer, trainer = lora_tuner.train_with_lora(
        # datasets["train"],
        # datasets["validation"],
        output_dir="./models/lora_adapters"
    )
    
    print("LoRA fine-tuning completed!")
```

### 2.4 Distributed Training for Large Models

**Multi-GPU Training Setup:**
```python
# scripts/distributed_training.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
from transformers import TrainingArguments
import os

class DistributedTrainer:
    def __init__(self):
        self.accelerator = Accelerator()
        
    def setup_distributed_training(self, model, train_dataloader, eval_dataloader, optimizer, lr_scheduler):
        """Setup distributed training with Accelerate"""
        
        # Prepare everything with accelerator
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        
        return model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    
    def create_training_args_for_distributed(self, output_dir):
        """Create training arguments optimized for distributed training"""
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            dataloader_num_workers=8,
            fp16=True,
            gradient_checkpointing=True,  # Save memory
            deepspeed="configs/deepspeed_config.json",  # Optional: DeepSpeed integration
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_steps=500,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            ddp_find_unused_parameters=False,
        )

# DeepSpeed configuration example
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    },
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}

# Save DeepSpeed config
import json
os.makedirs("configs", exist_ok=True)
## Phase 3: Model Optimization Techniques

### 3.1 Quantization Implementation

**Dynamic and Static Quantization:**
```python
# scripts/quantization.py
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_qconfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import os

class ModelQuantizer:
    def __init__(self, model_path, tokenizer_path=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
    def load_model(self):
        """Load the fine-tuned model"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # Start with float32 for quantization
            device_map="cpu"  # Load on CPU for quantization
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return model, tokenizer
    
    def dynamic_quantization(self, model, qconfig_spec=None):
        """Apply dynamic quantization (INT8)"""
        print("Applying dynamic quantization...")
        
        if qconfig_spec is None:
            # Default: quantize linear layers to INT8
            qconfig_spec = {nn.Linear}
        
        quantized_model = quantize_dynamic(
            model, 
            qconfig_spec, 
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def static_quantization(self, model, calibration_data):
        """Apply static quantization with calibration"""
        print("Applying static quantization...")
        
        # Set model to evaluation mode
        model.eval()
        
        # Prepare model for static quantization
        model.qconfig = default_qconfig
        torch.quantization.prepare(model, inplace=True)
        
        # Calibration phase
        print("Running calibration...")
        with torch.no_grad():
            for batch in calibration_data:
                model(**batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        return quantized_model
    
    def fp16_conversion(self, model):
        """Convert model to FP16 for memory efficiency"""
        print("Converting to FP16...")
        return model.half()
    
    def compare_models(self, original_model, quantized_model, test_input):
        """Compare original vs quantized model performance"""
        
        def measure_inference_time(model, input_data, num_runs=10):
            """Measure average inference time"""
            times = []
            model.eval()
            
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = model(**input_data)
                
                # Actual measurement
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(**input_data)
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            return sum(times) / len(times)
        
        def get_model_size(model):
            """Get model size in MB"""
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return (param_size + buffer_size) / 1024 / 1024  # Convert to MB
        
        # Measure original model
        original_time = measure_inference_time(original_model, test_input)
        original_size = get_model_size(original_model)
        
        # Measure quantized model
        quantized_time = measure_inference_time(quantized_model, test_input)
        quantized_size = get_model_size(quantized_model)
        
        # Calculate improvements
        speed_improvement = original_time / quantized_time
        size_reduction = original_size / quantized_size
        
        print(f"\n=== Model Comparison ===")
        print(f"Original Model:")
        print(f"  - Size: {original_size:.2f} MB")
        print(f"  - Inference Time: {original_time*1000:.2f} ms")
        print(f"Quantized Model:")
        print(f"  - Size: {quantized_size:.2f} MB")
        print(f"  - Inference Time: {quantized_time*1000:.2f} ms")
        print(f"Improvements:")
        print(f"  - Size Reduction: {size_reduction:.2f}x")
        print(f"  - Speed Improvement: {speed_improvement:.2f}x")
        
        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "original_time_ms": original_time * 1000,
            "quantized_time_ms": quantized_time * 1000,
            "size_reduction": size_reduction,
            "speed_improvement": speed_improvement
        }

# Advanced Quantization with BitsAndBytes
class AdvancedQuantizer:
    @staticmethod
    def load_4bit_model(model_name):
        """Load model with 4-bit quantization using BitsAndBytes"""
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        return model
    
    @staticmethod
    def load_8bit_model(model_name):
        """Load model with 8-bit quantization"""
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        return model

# Example usage
if __name__ == "__main__":
    # Initialize quantizer
    quantizer = ModelQuantizer("./models/finetuned")
    
    # Load original model
    model, tokenizer = quantizer.load_model()
    
    # Prepare test input
    test_text = "What is artificial intelligence?"
    test_input = tokenizer(test_text, return_tensors="pt", padding=True)
    
    # Apply dynamic quantization
    quantized_model = quantizer.dynamic_quantization(model)
    
    # Compare models
    results = quantizer.compare_models(model, quantized_model, test_input)
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), "./models/optimized/quantized_model.pth")
    
    print("Quantization completed and model saved!")
```

### 3.2 Model Pruning Implementation

**Structured and Unstructured Pruning:**
```python
# scripts/pruning.py
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM
import numpy as np
import copy

class ModelPruner:
    def __init__(self, model, pruning_ratio=0.2):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.original_model = copy.deepcopy(model)
    
    def unstructured_pruning(self, method="magnitude"):
        """Apply unstructured pruning to linear layers"""
        print(f"Applying unstructured pruning with {method} method...")
        
        # Get all linear layers
        modules_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_prune.append((module, 'weight'))
        
        # Apply pruning based on method
        if method == "magnitude":
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.pruning_ratio,
            )
        elif method == "random":
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=self.pruning_ratio,
            )
        
        return self.model
    
    def structured_pruning(self, dim=0):
        """Apply structured pruning (remove entire neurons/channels)"""
        print(f"Applying structured pruning...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune based on L2 norm of weights
                prune.ln_structured(
                    module, 
                    name="weight", 
                    amount=self.pruning_ratio, 
                    n=2, 
                    dim=dim
                )
        
        return self.model
    
    def gradual_pruning(self, num_steps=10):
        """Apply gradual pruning over multiple steps"""
        print(f"Applying gradual pruning over {num_steps} steps...")
        
        step_ratio = self.pruning_ratio / num_steps
        
        for step in range(num_steps):
            print(f"Pruning step {step + 1}/{num_steps}")
            
            # Apply small amount of pruning
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=step_ratio)
            
            # Optional: Fine-tune for a few steps here
            # self.fine_tune_step()
        
        return self.model
    
    def knowledge_distillation_pruning(self, teacher_model, student_model, distillation_alpha=0.7, temperature=4):
        """Combine pruning with knowledge distillation"""
        
        class DistillationLoss(nn.Module):
            def __init__(self, alpha=0.7, temperature=4):
                super().__init__()
                self.alpha = alpha
                self.temperature = temperature
                self.kl_div = nn.KLDivLoss(reduction="batchmean")
                self.ce_loss = nn.CrossEntropyLoss()
            
            def forward(self, student_outputs, teacher_outputs, labels):
                # Soft targets from teacher
                soft_teacher = nn.functional.softmax(teacher_outputs / self.temperature, dim=-1)
                soft_student = nn.functional.log_softmax(student_outputs / self.temperature, dim=-1)
                
                # Distillation loss
                distillation_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
                
                # Student loss on true labels
                student_loss = self.ce_loss(student_outputs, labels)
                
                # Combined loss
                total_loss = (self.alpha * distillation_loss + 
                             (1 - self.alpha) * student_loss)
                
                return total_loss
        
        return DistillationLoss(distillation_alpha, temperature)
    
    def calculate_sparsity(self):
        """Calculate the sparsity of the pruned model"""
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    total_params += mask.numel()
                    zero_params += (mask == 0).sum().item()
                else:
                    weights = module.weight
                    total_params += weights.numel()
                    zero_params += (weights == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        print(f"Model sparsity: {sparsity:.2%}")
        return sparsity
    
    def remove_pruning_masks(self):
        """Make pruning permanent by removing masks"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
        
        return self.model
    
    def compare_model_performance(self, test_dataloader, device='cuda'):
        """Compare original vs pruned model performance"""
        
        def evaluate_model(model, dataloader):
            model.eval()
            total_loss = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
                    
                    total_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)
            
            return total_loss / total_samples
        
        # Evaluate original model
        original_loss = evaluate_model(self.original_model, test_dataloader)
        
        # Evaluate pruned model
        pruned_loss = evaluate_model(self.model, test_dataloader)
        
        # Calculate degradation
        performance_degradation = (pruned_loss - original_loss) / original_loss * 100
        
        sparsity = self.calculate_sparsity()
        
        print(f"\n=== Pruning Results ===")
        print(f"Original Model Loss: {original_loss:.4f}")
        print(f"Pruned Model Loss: {pruned_loss:.4f}")
        print(f"Performance Degradation: {performance_degradation:.2f}%")
        print(f"Model Sparsity: {sparsity:.2%}")
        
        return {
            "original_loss": original_loss,
            "pruned_loss": pruned_loss,
            "degradation_percent": performance_degradation,
            "sparsity_percent": sparsity * 100
        }

# Example usage
if __name__ == "__main__":
    # Load your fine-tuned model
    model = AutoModelForCausalLM.from_pretrained("./models/finetuned")
    
    # Initialize pruner
    pruner = ModelPruner(model, pruning_ratio=0.3)
    
    # Apply unstructured pruning
    pruned_model = pruner.unstructured_pruning(method="magnitude")
    
    # Calculate sparsity
    sparsity = pruner.calculate_sparsity()
    
    # Make pruning permanent
    pruned_model = pruner.remove_pruning_masks()
    
    # Save pruned model
    torch.save(pruned_model.state_dict(), "./models/optimized/pruned_model.pth")
    
### 3.3 ONNX Conversion and TensorRT Optimization

**Model Conversion for Production Inference:**
```python
# scripts/onnx_tensorrt_optimization.py
import torch
import onnx
import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer
import tensorrt as trt
import numpy as np
import time

class ONNXConverter:
    def __init__(self, model_path, tokenizer_path=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
    def convert_to_onnx(self, output_path, optimize=True):
        """Convert PyTorch model to ONNX format"""
        print("Converting model to ONNX...")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input
        dummy_text = "This is a sample input for ONNX conversion."
        dummy_input = tokenizer(dummy_text, return_tensors="pt", padding=True, max_length=512)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=11,
            do_constant_folding=True,
            export_params=True,
        )
        
        if optimize:
            self.optimize_onnx_model(output_path)
        
        print(f"ONNX model saved to: {output_path}")
    
    def optimize_onnx_model(self, onnx_path):
        """Optimize ONNX model for inference"""
        from onnxruntime.tools import optimizer
        
        # Load ONNX model
        model = onnx.load(onnx_path)
        
        # Apply optimizations
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type='transformer',
            num_heads=12,  # Adjust based on your model
            hidden_size=768,  # Adjust based on your model
            optimization_level=99,  # Maximum optimization
        )
        
        # Save optimized model
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        optimized_model.save_model_to_file(optimized_path)
        
        print(f"Optimized ONNX model saved to: {optimized_path}")
        return optimized_path

class TensorRTOptimizer:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
    def build_tensorrt_engine(self, engine_path, precision='fp16', max_batch_size=1, max_seq_length=512):
        """Build TensorRT engine from ONNX model"""
        print(f"Building TensorRT engine with {precision} precision...")
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(self.onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            # Add INT8 calibrator here if needed
        
        # Set optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()
        profile.set_shape('input_ids', (1, 1), (max_batch_size, max_seq_length//2), (max_batch_size, max_seq_length))
        profile.set_shape('attention_mask', (1, 1), (max_batch_size, max_seq_length//2), (max_batch_size, max_seq_length))
        config.add_optimization_profile(profile)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return None
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to: {engine_path}")
        return engine_path
    
    def create_inference_session(self, engine_path):
        """Create TensorRT inference session"""
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        return engine, context

class InferenceComparison:
    """Compare inference performance across different optimization methods"""
    
    def __init__(self, pytorch_model_path, onnx_model_path=None, tensorrt_engine_path=None):
        self.pytorch_model_path = pytorch_model_path
        self.onnx_model_path = onnx_model_path
        self.tensorrt_engine_path = tensorrt_engine_path
        
    def benchmark_pytorch(self, test_inputs, num_runs=100):
        """Benchmark PyTorch model"""
        model = AutoModelForCausalLM.from_pretrained(self.pytorch_model_path)
        model.eval()
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                inputs = {k: v.to(device) for k, v in test_inputs.items()}
                _ = model(**inputs)
            
            # Benchmark
            torch.cuda.synchronize()
            for _ in range(num_runs):
                inputs = {k: v.to(device) for k, v in test_inputs.items()}
                
                start_time = time.time()
                _ = model(**inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return avg_time, std_time
    
    def benchmark_onnx(self, test_inputs, num_runs=100):
        """Benchmark ONNX model"""
        if self.onnx_model_path is None:
            return None, None
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(self.onnx_model_path, providers=providers)
        
        # Prepare inputs
        ort_inputs = {
            'input_ids': test_inputs['input_ids'].numpy(),
            'attention_mask': test_inputs['attention_mask'].numpy()
        }
        
        times = []
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, ort_inputs)
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, ort_inputs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return avg_time, std_time
    
    def run_full_comparison(self, test_text="What is artificial intelligence?"):
        """Run complete inference comparison"""
        # Prepare test inputs
        tokenizer = AutoTokenizer.from_pretrained(self.pytorch_model_path)
        test_inputs = tokenizer(test_text, return_tensors="pt", padding=True, max_length=512)
        
        results = {}
        
        # Benchmark PyTorch
        print("Benchmarking PyTorch model...")
        pytorch_time, pytorch_std = self.benchmark_pytorch(test_inputs)
        results['pytorch'] = {'avg_time': pytorch_time, 'std_time': pytorch_std}
        
        # Benchmark ONNX
        if self.onnx_model_path:
            print("Benchmarking ONNX model...")
            onnx_time, onnx_std = self.benchmark_onnx(test_inputs)
            results['onnx'] = {'avg_time': onnx_time, 'std_time': onnx_std}
        
        # Print comparison
        print(f"\n=== Inference Performance Comparison ===")
        for framework, metrics in results.items():
            if metrics['avg_time'] is not None:
                print(f"{framework.upper()}:")
                print(f"  Average time: {metrics['avg_time']*1000:.2f} ± {metrics['std_time']*1000:.2f} ms")
        
        # Calculate speedups
        if 'pytorch' in results and 'onnx' in results:
            if results['onnx']['avg_time'] is not None:
                speedup = results['pytorch']['avg_time'] / results['onnx']['avg_time']
                print(f"\nONNX Speedup: {speedup:.2f}x over PyTorch")
        
        return results

# Example usage
if __name__ == "__main__":
    model_path = "./models/finetuned"
    
    # Convert to ONNX
    converter = ONNXConverter(model_path)
    onnx_path = "./models/optimized/model.onnx"
    converter.convert_to_onnx(onnx_path, optimize=True)
    
    # Build TensorRT engine
    tensorrt_optimizer = TensorRTOptimizer(onnx_path)
    engine_path = "./models/optimized/model.trt"
    tensorrt_optimizer.build_tensorrt_engine(engine_path, precision='fp16')
    
    # Run inference comparison
    comparator = InferenceComparison(
        pytorch_model_path=model_path,
        onnx_model_path=onnx_path
    )
    results = comparator.run_full_comparison()
    
    print("Optimization pipeline completed!")
```

## Phase 4: Deployment Pipeline

### 4.1 FastAPI Model Serving

**Production-Ready API Server:**
```python
# deployment/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import asyncio
import time
import logging
import uvicorn
from contextlib import asynccontextmanager
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model loading
model = None
tokenizer = None
text_generator = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    max_length: int = Field(default=100, ge=10, le=500)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)

class ChatResponse(BaseModel):
    response: str
    confidence_score: float
    processing_time_ms: float
    model_info: Dict[str, Any]

class SystemStatus(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    model_loaded: bool
    uptime: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading model and tokenizer...")
    await load_model()
    logger.info("Model loaded successfully!")
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="LLM Fine-tuned Model API",
    description="Production API for fine-tuned Large Language Model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def load_model():
    """Load the fine-tuned model and tokenizer"""
    global model, tokenizer, text_generator
    
    try:
        model_path = "./models/finetuned"  # Update with your model path
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        # Create text generation pipeline
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        logger.info(f"Model loaded on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "LLM API is running"}

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Detailed system health check"""
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    gpu_usage = None
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = gpus[0].load * 100
    except:
        pass
    
    # Calculate uptime (simplified)
    uptime = "Running"
    
    return SystemStatus(
        cpu_usage=cpu_usage,
        memory_usage=memory.percent,
        gpu_usage=gpu_usage,
        model_loaded=model is not None,
        uptime=uptime
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate response for chat message"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Format input
        input_text = f"User: {request.message}\nAssistant:"
        
        # Generate response
        outputs = text_generator(
            input_text,
            max_length=len(tokenizer.encode(input_text)) + request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Extract response
        generated_text = outputs[0]['generated_text']
        response_text = generated_text[len(input_text):].strip()
        
        # Calculate confidence (simplified - you can implement more sophisticated methods)
        confidence_score = min(1.0, len(response_text) / request.max_length)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response=response_text,
            confidence_score=confidence_score,
            processing_time_ms=processing_time,
            model_info={
                "model_type": "fine-tuned-llm",
                "parameters": f"~{model.num_parameters():,}",
                "device": str(next(model.parameters()).device)
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/batch_chat")
async def batch_chat(requests: List[ChatRequest]):
    """Process multiple chat requests in batch"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    responses = []
    
    for req in requests:
        try:
            response = await chat(req)
            responses.append(response)
        except Exception as e:
            responses.append({
                "error": str(e),
                "original_request": req.dict()
            })
    
    return {"responses": responses}

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "parameters": f"{model.num_parameters():,}",
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "tokenizer_vocab_size": len(tokenizer.get_vocab()),
        "max_position_embeddings": getattr(model.config, 'max_position_embeddings', 'Unknown')
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 worker for GPU models
        reload=False
    )
```

### 4.2 Docker Containerization

**Multi-Stage Dockerfile for Production:**
```dockerfile
# deployment/docker/Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Build environment
FROM pytorch/pytorch:2.1.0-cuda11.8-devel as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Stage 2: Production runtime
FROM pytorch/pytorch:2.1.0-cuda11.8-runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy application code
COPY --from=builder /app /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Phase 5: Evaluation and Testing

### 5.1 Comprehensive Model Evaluation

**Performance Metrics and Benchmarking:**
```python
# scripts/evaluation.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
import json
import pandas as pd
from typing import List, Dict, Any
import logging

class LLMEvaluator:
    def __init__(self, model_path, tokenizer_path=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.model = None
        self.tokenizer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_model(self):
        """Load model and tokenizer for evaluation"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
    def generate_responses(self, prompts: List[str], max_length: int = 100) -> List[str]:
        """Generate responses for a list of prompts"""
        responses = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=inputs.input_ids.shape[1] + max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            responses.append(response.strip())
            
        return responses
    
    def calculate_perplexity(self, test_texts: List[str]) -> float:
        """Calculate perplexity on test set"""
        total_loss = 0
        total_tokens = 0
        
        for text in test_texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                
            total_loss += loss.item() * inputs.input_ids.size(1)
            total_tokens += inputs.input_ids.size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return perplexity.item()
    
    def run_comprehensive_evaluation(self, test_dataset: Dataset) -> Dict[str, Any]:
        """Run comprehensive evaluation on test dataset"""
        print("Starting comprehensive evaluation...")
        
        # Extract prompts and references
        prompts = [item['input_text'] for item in test_dataset]
        references = [item['target_text'] for item in test_dataset]
        test_texts = [item['full_text'] for item in test_dataset]
        
        # Generate predictions
        print("Generating predictions...")
        predictions = self.generate_responses(prompts)
        
        # Calculate metrics
        print("Calculating metrics...")
        
        # Perplexity
        perplexity = self.calculate_perplexity(test_texts[:100])  # Limit for speed
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        
        # BERT scores
        bert_scores = self.calculate_bert_scores(predictions, references)
        
        # Coherence
        coherence = self.evaluate_coherence(predictions)
        
        # Compile results
        results = {
            'perplexity': perplexity,
            'coherence': coherence,
            **rouge_scores,
            **bert_scores,
            'num_samples': len(test_dataset)
        }
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = LLMEvaluator("./models/finetuned")
    evaluator.load_model()
    
    # Run evaluation
    # results = evaluator.run_comprehensive_evaluation(test_dataset)
    
    print("Evaluation completed!")
```

## Project Milestones and Timeline

### Week 1-2: Foundation Setup
- ✅ Environment setup and dependency installation
- ✅ Data preparation and preprocessing pipeline
- ✅ Model selection and baseline establishment
- ✅ Initial fine-tuning experiments

### Week 3-4: Advanced Fine-tuning
- ✅ LoRA/QLoRA implementation for efficiency
- ✅ Distributed training setup for large models
- ✅ Hyperparameter optimization
- ✅ Model evaluation and validation

### Week 5: Optimization Implementation
- ✅ Quantization techniques (INT8, FP16, 4-bit)
- ✅ Model pruning (structured and unstructured)
- ✅ Knowledge distillation for compression
- ✅ ONNX and TensorRT conversion

### Week 6: Deployment and Production
- ✅ FastAPI server development
- ✅ Docker containerization
- ✅ Kubernetes deployment setup
- ✅ Monitoring and observability
- ✅ Performance benchmarking

## Success Metrics

### Technical Performance
- **Perplexity Improvement**: >15% reduction from baseline
- **Inference Speed**: >3x speedup with optimization
- **Model Size**: >50% reduction through quantization/pruning
- **API Response Time**: <500ms for typical queries
- **System Uptime**: >99.5% availability

### Learning Outcomes
- **Deep understanding** of LLM architecture and fine-tuning
- **Practical experience** with production ML systems
- **Optimization expertise** for memory and speed constraints
- **Deployment skills** for scalable AI applications
- **Monitoring capabilities** for ML system health

## Next Steps and Extensions

### Advanced Features
- **Multi-modal capabilities** (text + image processing)
- **Retrieval-augmented generation** (RAG) integration
- **Tool usage and function calling** capabilities
- **Multi-language support** and cross-lingual transfer
- **Federated learning** for privacy-preserving training

### Production Enhancements
- **Auto-scaling** based on traffic patterns
- **A/B testing** framework for model comparison
- **Continuous training** pipeline with new data
- **Model versioning** and rollback capabilities
- **Advanced security** and privacy features

## Resources and Further Learning

### Books
- "Attention Is All You Need" - Transformer architecture
- "The Illustrated Transformer" - Visual guide to transformers
- "Building LLMs for Production" - Practical deployment guide

### Online Courses
- Hugging Face NLP Course
- CS224N: Natural Language Processing with Deep Learning
- Deep Learning Specialization (Coursera)

### Tools and Frameworks
- **Hugging Face Hub**: Model sharing and collaboration
- **Weights & Biases**: Experiment tracking
- **MLflow**: ML lifecycle management
- **Ray**: Distributed computing for ML
- **Kubeflow**: ML workflows on Kubernetes

---

**🎉 Congratulations!** You've completed a comprehensive LLM fine-tuning and optimization project. You now have the skills to fine-tune, optimize, and deploy large language models in production environments!
