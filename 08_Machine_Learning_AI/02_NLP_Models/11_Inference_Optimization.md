# Inference Optimization

## Topics
- Inference techniques like KV caching
- Batching strategies
- Speculative decoding and other optimizations
- Efficient inference pipelines

### Example: Inference with KV Caching (Python)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, use_cache=True)
print(tokenizer.decode(outputs[0]))
```
