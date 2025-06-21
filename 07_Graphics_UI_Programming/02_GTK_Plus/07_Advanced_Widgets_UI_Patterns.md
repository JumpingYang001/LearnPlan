# Advanced Widgets and UI Patterns

## Complex Widgets
- GtkHeaderBar
- GtkSearchBar
- GtkPopover
- GtkRevealer
- GtkScrolledWindow
- GtkOverlay

## Application UI Patterns
- GtkApplication
- Application window
- Action system
- Menus and toolbars
- Keyboard shortcuts
- UI definition files

## Drag and Drop
- Drag sources
- Drop targets
- Data transfer
- Custom formats

## Accessibility
- ATK/AT-SPI integration
- Accessible roles and states
- Screen reader support
- Keyboard navigation

### Example: GtkHeaderBar
```c
GtkWidget *header = gtk_header_bar_new();
gtk_header_bar_set_title(GTK_HEADER_BAR(header), "My App");
```
