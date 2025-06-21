# Qt Multimedia and Sensors

## Audio and Video
- MediaPlayer element
- Audio output
- Video playback
- Camera access
- Media recording

## Sensor Integration
- Accelerometer
- Gyroscope
- Ambient light
- Proximity
- Position and GPS

### Example: MediaPlayer in QML
```qml
import QtMultimedia 5.15

MediaPlayer {
    id: player
    source: "music.mp3"
    autoPlay: true
}
```
