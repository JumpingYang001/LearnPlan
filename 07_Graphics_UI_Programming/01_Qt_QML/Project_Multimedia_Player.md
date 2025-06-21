# Project: Multimedia Player

## Objective
Create a media player with custom controls, supporting various media formats and playlist management.

## Key Features
- Audio/video playback
- Custom play/pause/seek controls
- Playlist management
- QML-based UI

## Example: QML Media Player
```qml
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15

ApplicationWindow {
    visible: true
    width: 640; height: 480
    MediaPlayer { id: player; source: "song.mp3" }
    Button { text: "Play"; onClicked: player.play() }
    Button { text: "Pause"; onClicked: player.pause() }
}
```
