# SOA Fundamentals

## Description
Covers SOA principles, components, service contracts, interfaces, ESB concepts, and comparison with other architectural styles.

## Example
```xml
<!-- Example of a simple SOA service contract in WSDL -->
<definitions name="HelloService"
             targetNamespace="http://www.examples.com/wsdl/HelloService.wsdl"
             xmlns:tns="http://www.examples.com/wsdl/HelloService.wsdl"
             xmlns:xsd="http://www.w3.org/2001/XMLSchema"
             xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/"
             xmlns="http://schemas.xmlsoap.org/wsdl/">
  <message name="SayHelloRequest">
    <part name="firstName" type="xsd:string"/>
  </message>
  <message name="SayHelloResponse">
    <part name="greeting" type="xsd:string"/>
  </message>
  <portType name="Hello_PortType">
    <operation name="sayHello">
      <input message="tns:SayHelloRequest"/>
      <output message="tns:SayHelloResponse"/>
    </operation>
  </portType>
</definitions>
```
