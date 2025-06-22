# Project: GEM300-Compliant Equipment Controller

## Description
Build a controller for 300mm processing equipment implementing all required GEM300 capabilities. Includes simulation for testing and validation.

## Example Code
```python
class EquipmentController:
    def __init__(self):
        self.state = 'IDLE'
    def start(self):
        self.state = 'RUNNING'
    def stop(self):
        self.state = 'STOPPED'

# Simulation
controller = EquipmentController()
controller.start()
print(controller.state)  # Output: RUNNING
```
