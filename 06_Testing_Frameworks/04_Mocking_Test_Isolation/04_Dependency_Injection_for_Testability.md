# Dependency Injection for Testability

## Description
Dependency injection (DI) is a technique for providing dependencies to a class, making it easier to substitute test doubles during testing.

## Example (C#)
```csharp
public class Calculator {
    private readonly ILogger _logger;
    public Calculator(ILogger logger) {
        _logger = logger;
    }
}
```
