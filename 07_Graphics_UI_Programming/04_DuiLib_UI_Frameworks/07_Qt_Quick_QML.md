# Qt Quick and QML

## Topics
- Qt Quick architecture
- QML language and syntax
- Integration with C++
- Comparison with DuiLib approach

### Example: Simple QML UI
```qml
import QtQuick 2.15
Rectangle {
    width: 200; height: 100
    color: "#f0f0f0"
    Text { text: "Hello QML!"; anchors.centerIn: parent }
}
```
