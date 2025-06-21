# GTK+ CSS Styling

## GTK+ CSS System
- CSS selector syntax
- Style properties
- Theme integration
- Widget states

## Styling Techniques
- Inline CSS
- CSS provider
- Style contexts
- Style classes
- Loading from file

## Custom Drawing
- Drawing areas
- Cairo integration
- Custom widget rendering
- Canvas widgets

### Example: Applying CSS
```c
GtkCssProvider *provider = gtk_css_provider_new();
gtk_css_provider_load_from_data(provider, "button { background: #3498db; }", -1);
gtk_style_context_add_provider_for_display(
    gdk_display_get_default(),
    GTK_STYLE_PROVIDER(provider),
    GTK_STYLE_PROVIDER_PRIORITY_USER);
```
