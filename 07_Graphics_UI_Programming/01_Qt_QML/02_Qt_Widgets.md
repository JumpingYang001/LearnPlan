# Qt Widgets

## Basic Widgets
- QLabel, QPushButton, QLineEdit
- QCheckBox, QRadioButton
- QComboBox, QListWidget
- QTableWidget, QTreeWidget
- QTabWidget, QStackedWidget

## Layout Management
- QHBoxLayout, QVBoxLayout
- QGridLayout, QFormLayout
- Layout nesting and spacing
- Size policies and constraints
- Splitters and docking

## Dialog Boxes
- Modal vs. non-modal dialogs
- Standard dialogs
- Custom dialog creation
- Dialog buttons
- Data exchange with dialogs

## Model/View Architecture
- Model concepts
- View widgets
- Delegates
- Custom models
- Sorting and filtering

## Styling and Appearance
- Widget style customization
- Style sheets
- Theme integration
- Platform-specific styling

### Example: QPushButton
```cpp
QPushButton *button = new QPushButton("Click Me");
button->setStyleSheet("background-color: lightblue;");
```
