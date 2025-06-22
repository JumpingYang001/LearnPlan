# Integration Testing with Partial Mocking

## Description
Partial mocking allows you to mock only certain methods of a class, enabling a mix of real and mocked behavior for integration tests.

## Example (Java)
```java
Calculator calc = spy(new Calculator());
doReturn(5).when(calc).add(2, 3);
assertEquals(5, calc.add(2, 3));
```
