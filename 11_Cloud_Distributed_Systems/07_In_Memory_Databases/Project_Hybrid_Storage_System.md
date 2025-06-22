# Project: Hybrid Storage System

## Description
Build a system combining in-memory and disk storage, implement tiered storage policies, and create data lifecycle management.

## Example Code
```python
# Example: Hybrid storage (memory + disk)
import pickle
memory_store = {}
def save_to_disk(key, value):
    with open(f'{key}.pkl', 'wb') as f:
        pickle.dump(value, f)
def get_data(key):
    if key in memory_store:
        return memory_store[key]
    try:
        with open(f'{key}.pkl', 'rb') as f:
            value = pickle.load(f)
            memory_store[key] = value
            return value
    except FileNotFoundError:
        return None
```
