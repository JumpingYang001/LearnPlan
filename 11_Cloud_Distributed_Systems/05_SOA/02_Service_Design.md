# Service Design in SOA

## Description
Discusses service granularity, business vs. technical services, reusability, composability, and versioning strategies.

## Example
```xml
<!-- Example of service versioning in WSDL -->
<definitions name="OrderService_v2"
             targetNamespace="http://www.example.com/wsdl/OrderService_v2.wsdl"
             xmlns:tns="http://www.example.com/wsdl/OrderService_v2.wsdl"
             xmlns:xsd="http://www.w3.org/2001/XMLSchema"
             xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/"
             xmlns="http://schemas.xmlsoap.org/wsdl/">
  <!-- ... service definition ... -->
</definitions>
```
