# QML Animation and Effects

## Animation Types
- PropertyAnimation
- NumberAnimation
- ColorAnimation
- RotationAnimation
- ParallelAnimation and SequentialAnimation

## Transitions
- State transitions
- Add/remove transitions
- Transition customization
- Easing curves

## States and State Management
- Defining states
- State transitions
- State groups
- PropertyChanges

## Visual Effects
- Gradients and shadows
- Opacity and opacity masks
- Blend modes
- Particle effects
- Shader effects

### Example: NumberAnimation
```qml
Rectangle {
    width: 100; height: 100
    color: "red"
    NumberAnimation on width { to: 300; duration: 1000 }
}
```
