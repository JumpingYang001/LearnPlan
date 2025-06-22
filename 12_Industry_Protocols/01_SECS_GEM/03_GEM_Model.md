# Generic Equipment Model (GEM)

## Description
Overview of the GEM (SEMI E30) standard, equipment states, events, and GEM-compliant interfaces.

## Key Concepts
- GEM standard
- Equipment states and transitions
- Collection events and data variables
- GEM-compliant equipment interfaces

## Example
```python
# Example: Equipment state transition (pseudo-code)
class Equipment:
    def __init__(self):
        self.state = 'IDLE'
    def start(self):
        self.state = 'RUNNING'
    def stop(self):
        self.state = 'IDLE'

machine = Equipment()
machine.start()
print(machine.state)
```
