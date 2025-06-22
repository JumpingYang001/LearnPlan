# Project: Legacy System Modernization

## Description
Wrap legacy systems with service interfaces, implement service façades, and create an integration layer for modern applications.

## Example Code
```java
// Example of a service façade in Java
public class LegacySystemFacade {
    private LegacySystem legacy;
    public LegacySystemFacade(LegacySystem legacy) {
        this.legacy = legacy;
    }
    public String getCustomerData(String id) {
        // Adapt legacy call
        return legacy.fetchData(id);
    }
}
```
