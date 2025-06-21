# Project: Desktop Utility Application

## Objective
Create a file management tool using GTK+, implementing drag and drop support and using GIO for file operations.

## Key Features
- File browser UI
- Drag and drop file operations
- GIO-based file management
- Multi-pane interface

### Example: File Browser Skeleton
```c
GtkWidget *window = gtk_window_new();
GtkWidget *list = gtk_list_box_new();
gtk_window_set_child(GTK_WINDOW(window), list);
```
