# Flutter for Desktop

## Topics
- Flutter desktop support
- Widget-based UI development
- Cross-platform capabilities
- Comparison with Windows-specific frameworks

### Example: Flutter Desktop App (Dart)
```dart
import 'package:flutter/material.dart';
void main() => runApp(MyApp());
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('Flutter Desktop')),
        body: Center(child: Text('Hello, Flutter!')),
      ),
    );
  }
}
```
