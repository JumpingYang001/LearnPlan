# EtherNet/IP

## Explanation
EtherNet/IP is an industrial protocol based on Ethernet and the Common Industrial Protocol (CIP). It supports both implicit (real-time) and explicit messaging for device communication.

## Example
```python
# Example: EtherNet/IP Explicit Messaging (Python, using cpppo)
from cpppo.server.enip import client
with client.connector(host='192.168.1.10') as conn:
    for v in conn.read([ '@4/150/3' ]):
        print(v)
```
