# Project: GEM300 to MES Integration Gateway

## Description
Build a gateway between GEM300 equipment and MES, implementing bidirectional communication, data transformation, and mapping.

## Example Code
```python
class MESGateway:
    def send_to_mes(self, data):
        print(f"Sending to MES: {data}")
    def receive_from_mes(self):
        return "MES Response"

gateway = MESGateway()
gateway.send_to_mes({"event": "Start"})
```
