# CoAP (Constrained Application Protocol)

## Explanation
CoAP is a web transfer protocol for constrained devices, using a RESTful model similar to HTTP but optimized for low-power and lossy networks. It supports resource discovery and observation.

## Example
```python
# Example: CoAP Client (aiocoap)
import asyncio
from aiocoap import *
async def main():
    protocol = await Context.create_client_context()
    request = Message(code=GET, uri='coap://localhost/resource')
    response = await protocol.request(request).response
    print('Result: %s\n%r' % (response.code, response.payload))
asyncio.run(main())
```
