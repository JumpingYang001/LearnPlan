# Event Handling and Signals

## Event System
- Event types
- Event propagation
- Event controllers (GTK4)
- Gesture recognition
- Key events
- Mouse events
- Touch events

## Signal Connection Methods
- g_signal_connect and variants
- Disconnect and blocking
- Default handlers
- Signal emission order
- Signal accumulation

### Example: Signal Connection
```c
g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
```
