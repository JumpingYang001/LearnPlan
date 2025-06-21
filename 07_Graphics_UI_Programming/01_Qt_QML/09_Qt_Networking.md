# Networking in Qt

## HTTP Communication
- QNetworkAccessManager
- GET/POST requests
- Authentication
- Cookie handling
- SSL/TLS support

## WebSockets
- QWebSocket
- Client implementation
- Server implementation
- Binary vs. text messages

## RESTful API Integration
- JSON parsing
- API client design
- Authentication handling
- Error handling
- Async operations

## Local Network Discovery
- QNetworkDatagram
- Multicast
- Service discovery
- Zero-configuration networking

### Example: QNetworkAccessManager
```cpp
QNetworkAccessManager *manager = new QNetworkAccessManager(this);
connect(manager, &QNetworkAccessManager::finished, this, &MyClass::replyFinished);
manager->get(QNetworkRequest(QUrl("https://api.example.com/data")));
```
