# HTTP/HTTPS for IoT

## Explanation
HTTP/HTTPS is widely used in IoT for RESTful APIs and webhooks. While heavier than MQTT/CoAP, it is suitable for devices with more resources and for interoperability with web services.

## Example
```python
# Example: HTTP POST with requests
import requests
response = requests.post('https://api.example.com/iot', json={'data': 42})
print(response.status_code)
```
