# Security in DDS

## Explanation
This section covers the DDS Security specification, including authentication, encryption, access control, secure discovery, and key distribution.

## Example Code (Pseudocode)
```cpp
// Pseudocode for enabling security
DDSParticipantQos qos;
qos.property.value["dds.sec.auth.plugin"] = "builtin.PKI-DH";
qos.property.value["dds.sec.access.plugin"] = "builtin.Access-Permissions";
participant.set_qos(qos);
```
