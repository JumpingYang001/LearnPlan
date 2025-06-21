# Project: Data Visualization Tool

## Objective
Build a chart/graph display application using GTK+ and Cairo for custom drawing, with interactive elements.

## Key Features
- Chart/graph rendering with Cairo
- Interactive data points
- Dynamic updates
- Custom widgets

### Example: Drawing a Bar Chart with Cairo
```c
cairo_t *cr = gdk_cairo_create(gtk_widget_get_window(widget));
cairo_set_source_rgb(cr, 0.2, 0.6, 0.8);
cairo_rectangle(cr, 20, 20, 100, 200);
cairo_fill(cr);
cairo_destroy(cr);
```
