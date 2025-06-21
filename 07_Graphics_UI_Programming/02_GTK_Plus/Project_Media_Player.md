# Project: Media Player

## Objective
Create a simple media player with GStreamer, a modern UI with HeaderBar, and playlist management.

## Key Features
- Audio/video playback with GStreamer
- Custom HeaderBar UI
- Playlist management
- Media controls

### Example: GStreamer Playback
```c
GstElement *player = gst_element_factory_make("playbin", "player");
g_object_set(player, "uri", "file:///path/to/media.mp3", NULL);
gst_element_set_state(player, GST_STATE_PLAYING);
```
