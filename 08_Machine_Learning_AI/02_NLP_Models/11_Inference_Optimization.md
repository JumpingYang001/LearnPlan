# Inference Optimization

*Duration: 3 weeks*

## Overview

Inference optimization is crucial for deploying NLP models in production environments where latency, throughput, and resource efficiency are critical. This section covers advanced inference techniques including KV caching, batching strategies, speculative decoding, and building efficient inference pipelines.

## 1. Key-Value Caching and Memory Optimization

### Understanding KV Caching

```python
import torch
import torch.nn as nn
import numpy as np
import time
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    AutoConfig
)
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

class KVCacheOptimizer:
    """Advanced KV caching and memory optimization for transformer inference"""
    
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.cache_stats = {}
    
    def generate_with_kv_cache(self, prompt, max_new_tokens=50, use_cache=True):
        """Generate text with KV caching enabled/disabled for comparison"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                do_sample=False,  # Deterministic for comparison
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'text': generated_text,
            'generation_time': generation_time,
            'use_cache': use_cache,
            'tokens_generated': max_new_tokens
        }
    
    def benchmark_kv_caching(self, prompts, max_new_tokens=50):
        """Benchmark KV caching vs non-caching performance"""
        
        results = {'with_cache': [], 'without_cache': []}
        
        for prompt in prompts:
            # With cache
            result_with_cache = self.generate_with_kv_cache(
                prompt, max_new_tokens, use_cache=True
            )
            results['with_cache'].append(result_with_cache)
            
            # Without cache
            result_without_cache = self.generate_with_kv_cache(
                prompt, max_new_tokens, use_cache=False
            )
            results['without_cache'].append(result_without_cache)
        
        # Calculate statistics
        with_cache_times = [r['generation_time'] for r in results['with_cache']]
        without_cache_times = [r['generation_time'] for r in results['without_cache']]
        
        speedup = np.mean(without_cache_times) / np.mean(with_cache_times)
        
        benchmark_results = {
            'results': results,
            'avg_time_with_cache': np.mean(with_cache_times),
            'avg_time_without_cache': np.mean(without_cache_times),
            'speedup': speedup,
            'cache_efficiency': (speedup - 1) * 100  # Percentage improvement
        }
        
        return benchmark_results
    
    def implement_custom_kv_cache(self, max_seq_length=512, num_layers=12, 
                                 num_heads=12, head_dim=64):
        """Implement custom KV cache management"""
        
        class CustomKVCache:
            def __init__(self, batch_size, num_layers, num_heads, head_dim, max_seq_length, device):
                self.batch_size = batch_size
                self.num_layers = num_layers
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.max_seq_length = max_seq_length
                self.device = device
                
                # Initialize cache tensors
                self.key_cache = torch.zeros(
                    num_layers, batch_size, num_heads, max_seq_length, head_dim,
                    dtype=torch.float16, device=device
                )
                self.value_cache = torch.zeros(
                    num_layers, batch_size, num_heads, max_seq_length, head_dim,
                    dtype=torch.float16, device=device
                )
                
                self.cache_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            def update(self, layer_idx, key_states, value_states, position):
                """Update cache with new key-value states"""
                batch_size, num_heads, seq_len, head_dim = key_states.shape
                
                # Update cache
                self.key_cache[layer_idx, :batch_size, :, position:position+seq_len] = key_states
                self.value_cache[layer_idx, :batch_size, :, position:position+seq_len] = value_states
                
                # Update cache lengths
                self.cache_lengths[:batch_size] = max(self.cache_lengths[:batch_size], position + seq_len)
            
            def get(self, layer_idx, batch_size):
                """Retrieve cached key-value states"""
                max_length = torch.max(self.cache_lengths[:batch_size]).item()
                
                return (
                    self.key_cache[layer_idx, :batch_size, :, :max_length],
                    self.value_cache[layer_idx, :batch_size, :, :max_length]
                )
            
            def clear(self):
                """Clear cache"""
                self.key_cache.zero_()
                self.value_cache.zero_()
                self.cache_lengths.zero_()
            
            def get_memory_usage(self):
                """Calculate memory usage in MB"""
                key_memory = self.key_cache.numel() * self.key_cache.element_size()
                value_memory = self.value_cache.numel() * self.value_cache.element_size()
                return (key_memory + value_memory) / (1024 * 1024)  # MB
        
        return CustomKVCache(1, num_layers, num_heads, head_dim, max_seq_length, self.device)
    
    def optimize_memory_usage(self, sequence_lengths, batch_sizes):
        """Optimize memory usage based on sequence patterns"""
        
        memory_analysis = {}
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                # Estimate memory requirements
                config = AutoConfig.from_pretrained(self.model_name)
                
                # KV cache memory calculation
                num_layers = config.n_layer
                num_heads = config.n_head
                head_dim = config.n_embd // config.n_head
                
                # Memory per token per layer (key + value)
                memory_per_token_per_layer = 2 * num_heads * head_dim * 2  # 2 bytes for fp16
                total_kv_memory = batch_size * seq_len * num_layers * memory_per_token_per_layer
                
                # Model parameters memory
                param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
                
                # Activation memory (rough estimate)
                activation_memory = batch_size * seq_len * config.n_embd * 4  # Rough estimate
                
                total_memory = total_kv_memory + param_memory + activation_memory
                
                memory_analysis[f"seq_{seq_len}_batch_{batch_size}"] = {
                    'kv_cache_memory_mb': total_kv_memory / (1024 * 1024),
                    'parameter_memory_mb': param_memory / (1024 * 1024),
                    'activation_memory_mb': activation_memory / (1024 * 1024),
                    'total_memory_mb': total_memory / (1024 * 1024),
                    'total_memory_gb': total_memory / (1024 * 1024 * 1024)
                }
        
        return memory_analysis
    
    def implement_sliding_window_cache(self, window_size=256):
        """Implement sliding window cache for long sequences"""
        
        class SlidingWindowCache:
            def __init__(self, window_size, num_layers, num_heads, head_dim, device):
                self.window_size = window_size
                self.num_layers = num_layers
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.device = device
                
                # Circular buffer for cache
                self.key_cache = torch.zeros(
                    num_layers, 1, num_heads, window_size, head_dim,
                    dtype=torch.float16, device=device
                )
                self.value_cache = torch.zeros(
                    num_layers, 1, num_heads, window_size, head_dim,
                    dtype=torch.float16, device=device
                )
                
                self.current_position = 0
                self.total_length = 0
            
            def update(self, layer_idx, key_states, value_states):
                """Update sliding window cache"""
                seq_len = key_states.shape[2]
                
                for i in range(seq_len):
                    pos = self.current_position % self.window_size
                    self.key_cache[layer_idx, 0, :, pos] = key_states[0, :, i]
                    self.value_cache[layer_idx, 0, :, pos] = value_states[0, :, i]
                    self.current_position += 1
                
                self.total_length += seq_len
            
            def get_effective_cache(self, layer_idx):
                """Get effective cache considering sliding window"""
                if self.total_length <= self.window_size:
                    # Haven't filled window yet
                    return (
                        self.key_cache[layer_idx, :, :, :self.current_position],
                        self.value_cache[layer_idx, :, :, :self.current_position]
                    )
                else:
                    # Reorder circular buffer
                    start_pos = self.current_position % self.window_size
                    
                    key_cache = torch.cat([
                        self.key_cache[layer_idx, :, :, start_pos:],
                        self.key_cache[layer_idx, :, :, :start_pos]
                    ], dim=2)
                    
                    value_cache = torch.cat([
                        self.value_cache[layer_idx, :, :, start_pos:],
                        self.value_cache[layer_idx, :, :, :start_pos]
                    ], dim=2)
                    
                    return key_cache, value_cache
            
            def get_memory_usage(self):
                """Get memory usage (constant due to sliding window)"""
                key_memory = self.key_cache.numel() * self.key_cache.element_size()
                value_memory = self.value_cache.numel() * self.value_cache.element_size()
                return (key_memory + value_memory) / (1024 * 1024)  # MB
        
        config = AutoConfig.from_pretrained(self.model_name)
        return SlidingWindowCache(
            window_size, config.n_layer, config.n_head,
            config.n_embd // config.n_head, self.device
        )
    
    def visualize_cache_performance(self, benchmark_results):
        """Visualize KV cache performance benefits"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Generation times comparison
        with_cache_times = [r['generation_time'] for r in benchmark_results['results']['with_cache']]
        without_cache_times = [r['generation_time'] for r in benchmark_results['results']['without_cache']]
        
        x = range(len(with_cache_times))
        ax1.bar([i - 0.2 for i in x], with_cache_times, 0.4, label='With Cache', alpha=0.7)
        ax1.bar([i + 0.2 for i in x], without_cache_times, 0.4, label='Without Cache', alpha=0.7)
        ax1.set_xlabel('Prompt Index')
        ax1.set_ylabel('Generation Time (seconds)')
        ax1.set_title('Generation Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup visualization
        speedups = [without / with for without, with in zip(without_cache_times, with_cache_times)]
        ax2.bar(x, speedups, color='green', alpha=0.7)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No Improvement')
        ax2.set_xlabel('Prompt Index')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('KV Cache Speedup by Prompt')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Memory usage estimation
        sequence_lengths = [50, 100, 200, 500, 1000]
        memory_usage = self.optimize_memory_usage(sequence_lengths, [1])
        
        kv_memories = [memory_usage[f"seq_{seq}_batch_1"]['kv_cache_memory_mb'] 
                      for seq in sequence_lengths]
        total_memories = [memory_usage[f"seq_{seq}_batch_1"]['total_memory_mb'] 
                         for seq in sequence_lengths]
        
        ax3.plot(sequence_lengths, kv_memories, 'o-', label='KV Cache Memory', linewidth=2)
        ax3.plot(sequence_lengths, total_memories, 's-', label='Total Memory', linewidth=2)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage vs Sequence Length')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cache efficiency
        avg_speedup = benchmark_results['speedup']
        cache_efficiency = benchmark_results['cache_efficiency']
        
        metrics = ['Speedup', 'Efficiency (%)']
        values = [avg_speedup, cache_efficiency]
        colors = ['skyblue', 'lightgreen']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Overall KV Cache Performance')
        ax4.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Example usage
def kv_cache_optimization_demo():
    """Demonstrate KV cache optimization techniques"""
    
    print("KV Cache Optimization Demo")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = KVCacheOptimizer("gpt2")
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Climate change poses significant challenges",
        "Advances in quantum computing will"
    ]
    
    print("\n1. Benchmarking KV Caching Performance...")
    benchmark_results = optimizer.benchmark_kv_caching(prompts, max_new_tokens=30)
    
    print(f"\nAverage time with cache: {benchmark_results['avg_time_with_cache']:.3f}s")
    print(f"Average time without cache: {benchmark_results['avg_time_without_cache']:.3f}s")
    print(f"Speedup: {benchmark_results['speedup']:.2f}x")
    print(f"Cache efficiency: {benchmark_results['cache_efficiency']:.1f}% improvement")
    
    print("\n2. Memory Usage Analysis...")
    memory_analysis = optimizer.optimize_memory_usage([128, 256, 512], [1, 4, 8])
    
    print("\nMemory usage for different configurations:")
    for config, usage in list(memory_analysis.items())[:3]:  # Show first 3
        print(f"  {config}:")
        print(f"    KV Cache: {usage['kv_cache_memory_mb']:.1f} MB")
        print(f"    Total: {usage['total_memory_mb']:.1f} MB")
    
    print("\n3. Custom KV Cache Implementation...")
    custom_cache = optimizer.implement_custom_kv_cache(max_seq_length=256)
    print(f"Custom cache memory usage: {custom_cache.get_memory_usage():.1f} MB")
    
    print("\n4. Sliding Window Cache...")
    sliding_cache = optimizer.implement_sliding_window_cache(window_size=128)
    print(f"Sliding window cache memory usage: {sliding_cache.get_memory_usage():.1f} MB")
    
    return optimizer, benchmark_results

# optimizer, benchmark_results = kv_cache_optimization_demo()
```

## 2. Batching Strategies and Dynamic Batching

### Advanced Batching Techniques

```python
import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time

class AdvancedBatchingSystem:
    """Advanced batching system for optimal inference throughput"""
    
    def __init__(self, model, tokenizer, max_batch_size=8, max_wait_time=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        # Request queue and batching state
        self.request_queue = queue.Queue()
        self.batch_stats = {
            'total_requests': 0,
            'total_batches': 0,
            'average_batch_size': 0,
            'wait_times': [],
            'processing_times': []
        }
        
        self.is_running = False
        self.batch_processor = None
    
    def add_request(self, text, max_length=50, **generation_kwargs):
        """Add a text generation request to the queue"""
        
        request = {
            'id': f"req_{int(time.time() * 1000000)}",
            'text': text,
            'max_length': max_length,
            'generation_kwargs': generation_kwargs,
            'timestamp': time.time(),
            'future': None
        }
        
        # Create a future for async result retrieval
        future = asyncio.Future()
        request['future'] = future
        
        self.request_queue.put(request)
        self.batch_stats['total_requests'] += 1
        
        return future
    
    def dynamic_batching(self):
        """Dynamic batching with adaptive batch sizes"""
        
        while self.is_running:
            batch = []
            batch_start_time = time.time()
            
            # Collect requests for batching
            while len(batch) < self.max_batch_size:
                try:
                    # Wait for requests with timeout
                    remaining_wait = self.max_wait_time - (time.time() - batch_start_time)
                    if remaining_wait <= 0 and batch:
                        break
                    
                    request = self.request_queue.get(timeout=max(0.001, remaining_wait))
                    batch.append(request)
                    
                except queue.Empty:
                    if batch:  # Process partial batch if we have requests
                        break
                    continue
            
            if batch:
                self.process_batch(batch)
    
    def process_batch(self, batch):
        """Process a batch of requests"""
        
        batch_start_time = time.time()
        
        try:
            # Prepare batch inputs
            texts = [req['text'] for req in batch]
            max_lengths = [req['max_length'] for req in batch]
            
            # Tokenize batch (pad to longest sequence)
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max(max_lengths)
            )
            
            # Generate for batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_lengths),  # Use minimum for efficiency
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode results
            results = []
            for i, output in enumerate(outputs):
                # Skip input tokens
                generated_tokens = output[inputs['input_ids'][i].shape[0]:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                results.append(generated_text)
            
            # Set results for futures
            for request, result in zip(batch, results):
                if not request['future'].done():
                    request['future'].set_result({
                        'generated_text': result,
                        'processing_time': time.time() - batch_start_time,
                        'wait_time': batch_start_time - request['timestamp']
                    })
        
        except Exception as e:
            # Set exception for all futures in failed batch
            for request in batch:
                if not request['future'].done():
                    request['future'].set_exception(e)
        
        # Update statistics
        processing_time = time.time() - batch_start_time
        self.batch_stats['total_batches'] += 1
        self.batch_stats['processing_times'].append(processing_time)
        self.batch_stats['average_batch_size'] = (
            (self.batch_stats['average_batch_size'] * (self.batch_stats['total_batches'] - 1) + len(batch)) /
            self.batch_stats['total_batches']
        )
        
        for request in batch:
            wait_time = batch_start_time - request['timestamp']
            self.batch_stats['wait_times'].append(wait_time)
    
    def adaptive_batch_sizing(self):
        """Adapt batch size based on system performance"""
        
        if len(self.batch_stats['processing_times']) < 10:
            return self.max_batch_size
        
        # Analyze recent performance
        recent_times = self.batch_stats['processing_times'][-10:]
        recent_sizes = [1] * len(recent_times)  # Simplified - would track actual sizes
        
        # Simple adaptive strategy: increase batch size if processing time per item decreases
        avg_time_per_item = np.mean(recent_times) / np.mean(recent_sizes)
        
        if avg_time_per_item < 0.1:  # If fast, try larger batches
            return min(self.max_batch_size * 2, 16)
        elif avg_time_per_item > 0.5:  # If slow, use smaller batches
            return max(self.max_batch_size // 2, 1)
        else:
            return self.max_batch_size
    
    def start_batch_processor(self):
        """Start the batch processing system"""
        
        self.is_running = True
        self.batch_processor = threading.Thread(target=self.dynamic_batching, daemon=True)
        self.batch_processor.start()
    
    def stop_batch_processor(self):
        """Stop the batch processing system"""
        
        self.is_running = False
        if self.batch_processor:
            self.batch_processor.join(timeout=5)
    
    def get_statistics(self):
        """Get batching system statistics"""
        
        stats = self.batch_stats.copy()
        
        if stats['wait_times']:
            stats['average_wait_time'] = np.mean(stats['wait_times'])
            stats['max_wait_time'] = np.max(stats['wait_times'])
        
        if stats['processing_times']:
            stats['average_processing_time'] = np.mean(stats['processing_times'])
            stats['throughput_requests_per_second'] = (
                stats['total_requests'] / sum(stats['processing_times']) 
                if sum(stats['processing_times']) > 0 else 0
            )
        
        return stats

class ContinuousBatchingSystem:
    """Continuous batching system inspired by vLLM"""
    
    def __init__(self, model, tokenizer, max_batch_size=32, max_seq_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        
        # Running batch state
        self.running_batch = []
        self.kv_cache = None
        self.sequence_lengths = {}
        
        self.batch_lock = threading.Lock()
        self.is_running = False
    
    def add_sequence(self, prompt, max_new_tokens=50):
        """Add a new sequence to the continuous batch"""
        
        sequence_id = f"seq_{len(self.sequence_lengths)}"
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        sequence_info = {
            'id': sequence_id,
            'prompt': prompt,
            'input_ids': inputs['input_ids'],
            'current_length': inputs['input_ids'].shape[1],
            'max_length': inputs['input_ids'].shape[1] + max_new_tokens,
            'is_finished': False,
            'generated_tokens': [],
            'start_time': time.time()
        }
        
        with self.batch_lock:
            if len(self.running_batch) < self.max_batch_size:
                self.running_batch.append(sequence_info)
                self.sequence_lengths[sequence_id] = sequence_info['current_length']
                return sequence_id
            else:
                return None  # Batch full
    
    def continuous_generation_step(self):
        """Perform one step of continuous batching generation"""
        
        with self.batch_lock:
            if not self.running_batch:
                return
            
            # Prepare batch inputs
            active_sequences = [seq for seq in self.running_batch if not seq['is_finished']]
            
            if not active_sequences:
                self.running_batch.clear()
                return
            
            # Get current input IDs for all active sequences
            batch_input_ids = []
            for seq in active_sequences:
                # For continuous generation, we only need the last token
                if seq['generated_tokens']:
                    current_input = torch.tensor([[seq['generated_tokens'][-1]]])
                else:
                    current_input = seq['input_ids']
                batch_input_ids.append(current_input)
            
            # Pad to same length
            max_len = max(ids.shape[1] for ids in batch_input_ids)
            padded_inputs = []
            attention_masks = []
            
            for input_ids in batch_input_ids:
                pad_length = max_len - input_ids.shape[1]
                if pad_length > 0:
                    padded_input = torch.cat([
                        torch.full((1, pad_length), self.tokenizer.pad_token_id),
                        input_ids
                    ], dim=1)
                    attention_mask = torch.cat([
                        torch.zeros(1, pad_length),
                        torch.ones(1, input_ids.shape[1])
                    ], dim=1)
                else:
                    padded_input = input_ids
                    attention_mask = torch.ones_like(input_ids)
                
                padded_inputs.append(padded_input)
                attention_masks.append(attention_mask)
            
            batch_inputs = torch.cat(padded_inputs, dim=0)
            batch_attention = torch.cat(attention_masks, dim=0)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_inputs,
                    attention_mask=batch_attention,
                    use_cache=True,
                    past_key_values=self.kv_cache
                )
                
                # Update KV cache
                self.kv_cache = outputs.past_key_values
                
                # Get next token predictions
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update sequences
            for i, seq in enumerate(active_sequences):
                next_token = next_tokens[i].item()
                seq['generated_tokens'].append(next_token)
                seq['current_length'] += 1
                
                # Check if sequence is finished
                if (next_token == self.tokenizer.eos_token_id or 
                    seq['current_length'] >= seq['max_length']):
                    seq['is_finished'] = True
                    seq['generated_text'] = self.tokenizer.decode(
                        seq['generated_tokens'], 
                        skip_special_tokens=True
                    )
                    seq['total_time'] = time.time() - seq['start_time']
            
            # Remove finished sequences
            self.running_batch = [seq for seq in self.running_batch if not seq['is_finished']]
    
    def run_continuous_batching(self, duration=10):
        """Run continuous batching for specified duration"""
        
        self.is_running = True
        start_time = time.time()
        step_count = 0
        
        while self.is_running and (time.time() - start_time) < duration:
            self.continuous_generation_step()
            step_count += 1
            time.sleep(0.01)  # Small delay to prevent CPU spinning
        
        return {
            'steps_completed': step_count,
            'duration': time.time() - start_time,
            'completed_sequences': [seq for seq in self.running_batch if seq['is_finished']]
        }

# Example usage and benchmarking
def batching_optimization_demo():
    """Demonstrate advanced batching techniques"""
    
    print("Advanced Batching Optimization Demo")
    print("=" * 50)
    
    # Load model for demo
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n1. Dynamic Batching System...")
    batch_system = AdvancedBatchingSystem(model, tokenizer, max_batch_size=4)
    batch_system.start_batch_processor()
    
    # Submit test requests
    test_prompts = [
        "The future of AI is",
        "Climate change affects",
        "Technology advances rapidly",
        "Machine learning enables",
        "Innovation drives progress"
    ]
    
    futures = []
    for prompt in test_prompts:
        future = batch_system.add_request(prompt, max_length=20)
        futures.append(future)
    
    # Wait for results
    results = []
    for future in futures:
        try:
            result = future.result(timeout=10)  # Wait up to 10 seconds
            results.append(result)
        except Exception as e:
            print(f"Request failed: {e}")
    
    batch_system.stop_batch_processor()
    
    # Display results
    print(f"Processed {len(results)} requests")
    for i, result in enumerate(results):
        print(f"  Request {i+1}:")
        print(f"    Generated: {result['generated_text'][:50]}...")
        print(f"    Wait time: {result['wait_time']:.3f}s")
        print(f"    Processing time: {result['processing_time']:.3f}s")
    
    # Get statistics
    stats = batch_system.get_statistics()
    print(f"\nBatching Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Average batch size: {stats['average_batch_size']:.2f}")
    if 'average_wait_time' in stats:
        print(f"  Average wait time: {stats['average_wait_time']:.3f}s")
    if 'throughput_requests_per_second' in stats:
        print(f"  Throughput: {stats['throughput_requests_per_second']:.2f} req/s")
    
    print("\n2. Continuous Batching System...")
    continuous_system = ContinuousBatchingSystem(model, tokenizer, max_batch_size=3)
    
    # Add sequences
    sequence_ids = []
    for prompt in test_prompts[:3]:  # Use first 3 for demo
        seq_id = continuous_system.add_sequence(prompt, max_new_tokens=15)
        if seq_id:
            sequence_ids.append(seq_id)
    
    print(f"Added {len(sequence_ids)} sequences to continuous batch")
    
    # Run continuous batching
    continuous_results = continuous_system.run_continuous_batching(duration=5)
    
    print(f"Continuous batching completed:")
    print(f"  Steps: {continuous_results['steps_completed']}")
    print(f"  Duration: {continuous_results['duration']:.2f}s")
    print(f"  Completed sequences: {len(continuous_results['completed_sequences'])}")
    
    return batch_system, continuous_system, results

# batch_system, continuous_system, results = batching_optimization_demo()
```

## 3. Speculative Decoding and Advanced Optimizations

### Speculative Decoding Implementation

```python
class SpeculativeDecoder:
    """Implementation of speculative decoding for faster generation"""
    
    def __init__(self, draft_model, target_model, tokenizer):
        """
        Args:
            draft_model: Smaller, faster model for generating draft sequences
            target_model: Larger, more accurate target model for verification
            tokenizer: Tokenizer for both models
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        
        self.draft_model.eval()
        self.target_model.eval()
        
        self.stats = {
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejection_points': [],
            'speedup_ratios': []
        }
    
    def generate_draft_sequence(self, input_ids, draft_length=5, temperature=1.0):
        """Generate draft sequence using the smaller model"""
        
        draft_tokens = []
        current_input = input_ids
        
        with torch.no_grad():
            for _ in range(draft_length):
                outputs = self.draft_model(current_input)
                logits = outputs.logits[0, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                draft_tokens.append(next_token.item())
                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=-1)
        
        return draft_tokens
    
    def verify_draft_sequence(self, input_ids, draft_tokens, temperature=1.0):
        """Verify draft sequence using the target model"""
        
        accepted_tokens = []
        rejection_point = len(draft_tokens)  # Default: accept all
        
        # Construct full sequence with draft tokens
        full_sequence = torch.cat([
            input_ids,
            torch.tensor(draft_tokens).unsqueeze(0)
        ], dim=-1)
        
        with torch.no_grad():
            # Get target model probabilities for the entire sequence
            outputs = self.target_model(full_sequence)
            target_logits = outputs.logits[0, -len(draft_tokens)-1:-1, :] / temperature
            target_probs = torch.softmax(target_logits, dim=-1)
            
            # Also get draft model probabilities for comparison
            draft_sequence = torch.cat([
                input_ids,
                torch.tensor(draft_tokens[:-1]).unsqueeze(0) if draft_tokens else input_ids[:, -1:]
            ], dim=-1)
            
            draft_outputs = self.draft_model(draft_sequence)
            draft_logits = draft_outputs.logits[0, -len(draft_tokens):, :] / temperature
            draft_probs = torch.softmax(draft_logits, dim=-1)
            
            # Verify each token
            for i, draft_token in enumerate(draft_tokens):
                target_prob = target_probs[i, draft_token].item()
                draft_prob = draft_probs[i, draft_token].item()
                
                # Acceptance probability
                acceptance_prob = min(1.0, target_prob / (draft_prob + 1e-10))
                
                if torch.rand(1).item() < acceptance_prob:
                    accepted_tokens.append(draft_token)
                else:
                    rejection_point = i
                    break
            
            # If we rejected a token, sample a new one from target distribution
            if rejection_point < len(draft_tokens):
                # Sample from adjusted distribution
                adjusted_probs = target_probs[rejection_point].clone()
                if rejection_point < len(draft_probs):
                    draft_prob_rejected = draft_probs[rejection_point, draft_tokens[rejection_point]]
                    adjusted_probs[draft_tokens[rejection_point]] = max(
                        0, adjusted_probs[draft_tokens[rejection_point]] - draft_prob_rejected
                    )
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                
                new_token = torch.multinomial(adjusted_probs, 1).item()
                accepted_tokens.append(new_token)
        
        return accepted_tokens, rejection_point
    
    def speculative_generate(self, prompt, max_new_tokens=50, draft_length=4, 
                           temperature=1.0):
        """Generate text using speculative decoding"""
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated_tokens = []
        
        start_time = time.time()
        
        while len(generated_tokens) < max_new_tokens:
            current_context = torch.cat([
                input_ids,
                torch.tensor(generated_tokens).unsqueeze(0) if generated_tokens else torch.empty(1, 0, dtype=torch.long)
            ], dim=-1)
            
            # Generate draft sequence
            remaining_tokens = max_new_tokens - len(generated_tokens)
            current_draft_length = min(draft_length, remaining_tokens)
            
            draft_tokens = self.generate_draft_sequence(
                current_context, 
                draft_length=current_draft_length,
                temperature=temperature
            )
            
            # Verify draft sequence
            accepted_tokens, rejection_point = self.verify_draft_sequence(
                current_context, 
                draft_tokens,
                temperature=temperature
            )
            
            # Update statistics
            self.stats['total_draft_tokens'] += len(draft_tokens)
            self.stats['accepted_tokens'] += len(accepted_tokens)
            self.stats['rejection_points'].append(rejection_point)
            
            # Add accepted tokens
            generated_tokens.extend(accepted_tokens)
            
            # Stop if we hit EOS token
            if self.tokenizer.eos_token_id in accepted_tokens:
                break
        
        generation_time = time.time() - start_time
        
        # Calculate theoretical speedup
        acceptance_rate = len(generated_tokens) / max(1, self.stats['total_draft_tokens'])
        theoretical_speedup = (draft_length * acceptance_rate + 1) / (draft_length + 1)
        self.stats['speedup_ratios'].append(theoretical_speedup)
        
        # Decode final text
        final_tokens = input_ids[0].tolist() + generated_tokens
        generated_text = self.tokenizer.decode(final_tokens, skip_special_tokens=True)
        
        return {
            'generated_text': generated_text,
            'generated_tokens': generated_tokens,
            'generation_time': generation_time,
            'acceptance_rate': acceptance_rate,
            'theoretical_speedup': theoretical_speedup
        }
    
    def benchmark_speculative_vs_standard(self, prompts, max_new_tokens=30):
        """Benchmark speculative decoding against standard generation"""
        
        results = {
            'speculative': [],
            'standard': [],
            'speedup_achieved': []
        }
        
        for prompt in prompts:
            # Speculative decoding
            spec_start = time.time()
            spec_result = self.speculative_generate(prompt, max_new_tokens)
            spec_time = time.time() - spec_start
            
            results['speculative'].append({
                'text': spec_result['generated_text'],
                'time': spec_time,
                'acceptance_rate': spec_result['acceptance_rate']
            })
            
            # Standard generation with target model
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            std_start = time.time()
            with torch.no_grad():
                outputs = self.target_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            std_time = time.time() - std_start
            
            std_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results['standard'].append({
                'text': std_text,
                'time': std_time
            })
            
            # Calculate actual speedup
            speedup = std_time / spec_time if spec_time > 0 else 1.0
            results['speedup_achieved'].append(speedup)
        
        return results
    
    def get_statistics(self):
        """Get speculative decoding statistics"""
        
        if self.stats['total_draft_tokens'] == 0:
            return self.stats
        
        stats = self.stats.copy()
        stats['overall_acceptance_rate'] = (
            self.stats['accepted_tokens'] / self.stats['total_draft_tokens']
        )
        stats['average_rejection_point'] = np.mean(self.stats['rejection_points'])
        stats['average_theoretical_speedup'] = np.mean(self.stats['speedup_ratios'])
        
        return stats

class ParallelDecodingOptimizer:
    """Advanced parallel decoding optimizations"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.parallel_stats = {}
    
    def beam_search_parallel(self, input_ids, num_beams=4, max_length=50):
        """Optimized parallel beam search"""
        
        batch_size = input_ids.shape[0]
        vocab_size = self.model.config.vocab_size
        
        # Initialize beams
        beam_scores = torch.zeros(batch_size, num_beams, device=input_ids.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        
        # Reshape for beam search
        input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        input_ids = input_ids.view(batch_size * num_beams, -1)
        
        beam_indices = torch.arange(batch_size * num_beams, device=input_ids.device)
        
        for step in range(max_length - input_ids.shape[1]):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
            
            # Convert to scores
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores
            next_token_scores = next_token_scores + beam_scores.view(-1, 1)
            
            # Reshape to (batch_size, num_beams * vocab_size)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top 2*num_beams scores
            next_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Calculate beam and token indices
            next_beam_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Select beams and update sequences
            beam_outputs = []
            for batch_idx in range(batch_size):
                batch_beam_indices = []
                batch_next_tokens = []
                
                for beam_token_rank in range(2 * num_beams):
                    beam_id = next_beam_indices[batch_idx, beam_token_rank]
                    token_id = next_tokens[batch_idx, beam_token_rank]
                    
                    if len(batch_beam_indices) < num_beams:
                        batch_beam_indices.append(beam_id.item())
                        batch_next_tokens.append(token_id.item())
                
                beam_outputs.append((batch_beam_indices, batch_next_tokens))
            
            # Update sequences
            new_input_ids = []
            new_beam_scores = torch.zeros(batch_size, num_beams, device=input_ids.device)
            
            for batch_idx in range(batch_size):
                beam_indices, next_tokens = beam_outputs[batch_idx]
                
                for new_beam_id, (beam_idx, token_id) in enumerate(zip(beam_indices, next_tokens)):
                    # Get original sequence
                    orig_idx = batch_idx * num_beams + beam_idx
                    orig_sequence = input_ids[orig_idx]
                    
                    # Append new token
                    new_sequence = torch.cat([orig_sequence, torch.tensor([token_id], device=input_ids.device)])
                    new_input_ids.append(new_sequence)
                    
                    # Update score
                    new_beam_scores[batch_idx, new_beam_id] = next_scores[batch_idx, beam_idx]
            
            input_ids = torch.stack(new_input_ids)
            beam_scores = new_beam_scores
            
            # Check for EOS tokens
            if self.tokenizer.eos_token_id in input_ids[:, -1]:
                break
        
        return input_ids, beam_scores
    
    def multi_gpu_generation(self, prompts, num_gpus=2):
        """Distribute generation across multiple GPUs"""
        
        if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
            print("Multi-GPU generation not available")
            return None
        
        # Split prompts across GPUs
        prompts_per_gpu = len(prompts) // num_gpus
        gpu_batches = []
        
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * prompts_per_gpu
            end_idx = start_idx + prompts_per_gpu if gpu_id < num_gpus - 1 else len(prompts)
            gpu_batches.append(prompts[start_idx:end_idx])
        
        # Create model replicas on different GPUs
        def generate_on_gpu(gpu_id, prompts_batch):
            device = f"cuda:{gpu_id}"
            model_replica = self.model.to(device)
            
            results = []
            for prompt in prompts_batch:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model_replica.generate(
                        **inputs,
                        max_new_tokens=30,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append(generated_text)
            
            return results
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, batch in enumerate(gpu_batches):
                future = executor.submit(generate_on_gpu, gpu_id, batch)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in futures:
                batch_results = future.result()
                all_results.extend(batch_results)
        
        return all_results

# Example usage and demonstration
def speculative_decoding_demo():
    """Demonstrate speculative decoding optimization"""
    
    print("Speculative Decoding Demo")
    print("=" * 50)
    
    # Load models (using same model as both draft and target for demo)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # In practice, draft_model would be smaller (e.g., GPT2-small vs GPT2-medium)
    draft_model = AutoModelForCausalLM.from_pretrained("gpt2")
    target_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize speculative decoder
    spec_decoder = SpeculativeDecoder(draft_model, target_model, tokenizer)
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Climate change is a global challenge",
        "Space exploration opens new frontiers"
    ]
    
    print("\n1. Speculative Generation Examples:")
    for i, prompt in enumerate(test_prompts):
        result = spec_decoder.speculative_generate(
            prompt, max_new_tokens=25, draft_length=4
        )
        
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generated: {result['generated_text'][len(prompt):].strip()}")
        print(f"Time: {result['generation_time']:.3f}s")
        print(f"Acceptance rate: {result['acceptance_rate']:.2%}")
        print(f"Theoretical speedup: {result['theoretical_speedup']:.2f}x")
    
    print("\n2. Benchmarking Speculative vs Standard Generation:")
    benchmark_results = spec_decoder.benchmark_speculative_vs_standard(
        test_prompts, max_new_tokens=20
    )
    
    print("\nComparison Results:")
    for i, (spec, std, speedup) in enumerate(zip(
        benchmark_results['speculative'],
        benchmark_results['standard'],
        benchmark_results['speedup_achieved']
    )):
        print(f"\nPrompt {i+1}:")
        print(f"  Speculative time: {spec['time']:.3f}s")
        print(f"  Standard time: {std['time']:.3f}s")
        print(f"  Actual speedup: {speedup:.2f}x")
        print(f"  Acceptance rate: {spec['acceptance_rate']:.2%}")
    
    # Overall statistics
    stats = spec_decoder.get_statistics()
    print(f"\nOverall Statistics:")
    print(f"  Total draft tokens: {stats['total_draft_tokens']}")
    print(f"  Accepted tokens: {stats['accepted_tokens']}")
    print(f"  Overall acceptance rate: {stats.get('overall_acceptance_rate', 0):.2%}")
    print(f"  Average theoretical speedup: {stats.get('average_theoretical_speedup', 1):.2f}x")
    
    return spec_decoder, benchmark_results

# spec_decoder, benchmark_results = speculative_decoding_demo()
```

## 4. Learning Objectives

By the end of this section, you should be able to:
- **Implement** KV caching and memory optimization techniques
- **Design** dynamic and continuous batching systems
- **Apply** speculative decoding for faster generation
- **Optimize** inference pipelines for production deployment
- **Benchmark** different optimization techniques
- **Deploy** efficient inference systems at scale

### Self-Assessment Checklist

□ Can implement and optimize KV caching systems  
□ Can design dynamic batching with adaptive sizing  
□ Can implement speculative decoding from scratch  
□ Can optimize memory usage for long sequences  
□ Can benchmark inference optimizations effectively  
□ Can deploy production-ready inference systems  
□ Can troubleshoot inference performance issues  

## 5. Practical Exercises

**Exercise 1: Custom Inference Engine**
```python
# TODO: Build a complete inference engine with KV caching
# Include dynamic batching, memory optimization, and monitoring
# Benchmark against standard implementations
```

**Exercise 2: Speculative Decoding System**
```python
# TODO: Implement speculative decoding with different draft models
# Optimize acceptance rates and measure real-world speedups
# Compare with other acceleration techniques
```

**Exercise 3: Production Inference Service**
```python
# TODO: Create a production-ready inference service
# Include batching, caching, monitoring, and auto-scaling
# Deploy with proper error handling and logging
```

## 6. Study Materials

### Essential Papers
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

### Optimization Resources
- **Libraries**: vLLM, TensorRT-LLM, DeepSpeed, FasterTransformer
- **Techniques**: KV caching, batching, quantization, pruning
- **Hardware**: CUDA, tensor cores, memory hierarchy
- **Profiling**: NVIDIA Nsight, PyTorch Profiler

### Tools and Libraries
```bash
pip install torch transformers accelerate
pip install vllm tensorrt
pip install flash-attn
pip install triton
```

## 7. Next Steps

**Next Section:** [Deployment Architectures](12_Deployment_Architectures.md)

**Related Topics:**
- Hardware acceleration (GPUs, TPUs)
- Quantization and compression
- Distributed inference
- Edge deployment optimization
