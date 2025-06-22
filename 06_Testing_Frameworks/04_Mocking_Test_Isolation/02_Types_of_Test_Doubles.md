# Types of Test Doubles

## Description
Test doubles are objects that stand in for real dependencies in tests. Common types include mocks, stubs, fakes, spies, and dummies. Each serves a different purpose in isolating the code under test.

## Example (Java)
```java
// Using Mockito for a mock
Calculator calc = mock(Calculator.class);
when(calc.add(2, 3)).thenReturn(5);
assertEquals(5, calc.add(2, 3));
```
