# Integration with Other Libraries

## Cairo Graphics
- Drawing context
- Paths and shapes
- Text rendering
- Image surfaces

## Pango Text Layout
- Text rendering
- Font handling
- Text attributes
- International text

## GStreamer Integration
- Media playback
- Audio/video widgets
- Pipeline integration

## WebKit Integration
- Web content display
- JavaScript interaction
- HTML rendering

### Example: Cairo Drawing
```c
cairo_t *cr = gdk_cairo_create(gtk_widget_get_window(widget));
cairo_set_source_rgb(cr, 0, 0, 1);
cairo_rectangle(cr, 10, 10, 100, 50);
cairo_fill(cr);
cairo_destroy(cr);
```
