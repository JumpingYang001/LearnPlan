# RPC Fundamentals

## Overview
Remote Procedure Call (RPC) is a protocol that allows a program to execute code on a remote system as if it were a local procedure call. It abstracts the communication between distributed components, making it easier to build scalable systems.

## Key Concepts
- RPC vs REST: RPC is more tightly coupled and efficient for internal service communication, while REST is stateless and widely used for web APIs.
- Serialization: Data is converted to a format suitable for transmission (e.g., JSON, binary).
- Service Discovery: Mechanisms to locate services in a distributed environment.

## C/C++ Example: Simple RPC Concept
```c
// Pseudo-code for a simple RPC call
int add(int a, int b) {
    // This could be a remote call in a real RPC framework
    return a + b;
}

int main() {
    int result = add(2, 3); // In RPC, this would call a remote service
    printf("Result: %d\n", result);
    return 0;
}
```
