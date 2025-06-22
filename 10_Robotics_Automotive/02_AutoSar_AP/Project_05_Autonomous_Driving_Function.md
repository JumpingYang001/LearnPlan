# Project 5: Autonomous Driving Function

## Description
Implement a basic autonomous driving function using Adaptive AUTOSAR. Integrate with perception, planning, and control modules. Ensure compliance with safety standards.

## Example Code
```cpp
// Perception, planning, and control (pseudo-code)
PerceptionData perception = perceptionModule.getData();
Plan plan = planningModule.createPlan(perception);
controlModule.execute(plan);
// Safety compliance
if (!safetyCheck(plan)) {
    controlModule.stop();
}
```
