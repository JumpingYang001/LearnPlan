# Equipment Engineering Capabilities (EEC)

## Description
Introduction to EEC (SEMI E164), equipment health monitoring, and fault detection.

## Key Concepts
- EEC standard
- Equipment health monitoring
- Fault detection and classification
- EEC features

## Example
```python
# Example: Fault detection (pseudo-code)
def detect_fault(equipment_data):
    if equipment_data['temperature'] > 100:
        return 'FAULT: Overheat'
    return 'OK'

result = detect_fault({'temperature': 120})
print(result)
```
