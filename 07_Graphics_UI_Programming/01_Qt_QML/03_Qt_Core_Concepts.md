# Qt Core Concepts

## Qt Object Model
- QObject hierarchy
- Object ownership
- Parent-child relationships
- Memory management

## Signals and Slots
- Connection syntax
- Signal emission
- Slot implementation
- Connection types
- Lambda connections

## Property System
- Q_PROPERTY macro
- Property accessors
- Property bindings
- Dynamic properties

## Event Handling
- Event types
- Event filters
- Event propagation
- Custom events

## Threading in Qt
- QThread usage
- Worker objects
- Thread safety
- Synchronization primitives
- Thread pools

### Example: Signal and Slot
```cpp
connect(button, &QPushButton::clicked, this, &MainWindow::onButtonClicked);
```
