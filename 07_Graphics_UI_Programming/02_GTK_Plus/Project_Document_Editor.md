# Project: Document Editor

## Objective
Build a text editor with syntax highlighting, file operations, printing, and search/replace functionality.

## Key Features
- Text editing widget
- Syntax highlighting
- File open/save
- Print support
- Search and replace

### Example: TextView Widget
```c
GtkWidget *textview = gtk_text_view_new();
GtkTextBuffer *buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(textview));
gtk_text_buffer_set_text(buffer, "Hello, GTK+ Editor!", -1);
```
