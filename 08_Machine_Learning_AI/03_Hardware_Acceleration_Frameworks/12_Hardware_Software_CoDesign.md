# Hardware-Software Co-design

## Topics
- HW-SW co-design for ML
- Algorithm-hardware mapping
- Compiler optimization for specific hardware
- Co-optimized ML solutions

### Example: HW-SW Co-design Simulation (Python)
```python
def sw_part(x):
    return x + 1

def hw_part(x):
    return x * 2

def co_design(x):
    return hw_part(sw_part(x))
print(co_design(3))
```
