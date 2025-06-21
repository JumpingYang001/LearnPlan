# QML Fundamentals

## QML Language Basics
- Syntax and structure
- Object declarations
- Property assignments
- JavaScript integration
- Comments and documentation

## Basic QML Types
- Item and Rectangle
- Text and Image
- MouseArea and input handling
- Positioning elements
- Anchors and layouts

## QML Component Organization
- Importing modules
- Creating components
- File structure
- Component reuse
- Namespaces

## Property Binding
- Declarative bindings
- Binding expressions
- One-way vs. two-way binding
- Binding loops and debugging

## QML/C++ Integration
- Registering C++ types
- Property exposure
- Invoking QML from C++
- Invoking C++ from QML
- Context properties

### Example: Simple QML Rectangle
```qml
import QtQuick 2.15

Rectangle {
    width: 200; height: 100
    color: "lightgreen"
    Text { text: "Hello QML!"; anchors.centerIn: parent }
}
```
