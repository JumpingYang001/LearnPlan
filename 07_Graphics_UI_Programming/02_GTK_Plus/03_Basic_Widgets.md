# Basic GTK+ Widgets

## Window and Container Widgets
- GtkWindow
- GtkBox (horizontal/vertical)
- GtkGrid
- GtkPaned
- GtkNotebook
- GtkStack and GtkStackSwitcher

## Basic Control Widgets
- GtkButton
- GtkLabel
- GtkEntry
- GtkCheckButton
- GtkToggleButton
- GtkSwitch
- GtkSpinButton

## Selection Widgets
- GtkDropDown (GTK4) / GtkComboBox (GTK3)
- GtkListBox
- GtkFlowBox
- GtkTreeView (for complex data)

## Dialog Widgets
- GtkDialog
- GtkMessageDialog
- GtkFileChooserDialog
- GtkColorChooserDialog
- GtkFontChooserDialog

## Layout and Positioning
- Box layout
- Grid layout
- Widget alignment
- Margin and padding
- Constraint-based layout (GTK4)

### Example: GtkButton
```c
GtkWidget *button = gtk_button_new_with_label("Click Me");
g_signal_connect(button, "clicked", G_CALLBACK(on_button_clicked), NULL);
```
