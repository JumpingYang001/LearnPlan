# GTK+ Introduction and Setup

## GTK+ Overview
- History and evolution
- GTK3 vs. GTK4
- Object-oriented design
- GObject system
- Language bindings

## Development Environment Setup
- Installing GTK+ development packages
- Development tools (GCC, pkg-config)
- IDE integration (VS Code, GNOME Builder)
- Build systems (Meson, CMake)

## First GTK+ Application
- Basic window creation
- Application lifecycle
- Event loop
- Signal connections
- Hello World example

### Example: Hello World (C)
```c
#include <gtk/gtk.h>

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);
    GtkWidget *window = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window), "Hello, GTK+");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 200);
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    gtk_widget_show(window);
    gtk_main();
    return 0;
}
```
