# Mockito for Java/Kotlin

## Description
Mockito is a popular Java/Kotlin mocking framework that allows you to create and configure mock objects for testing.

## Example (Java)
```java
Calculator calc = mock(Calculator.class);
when(calc.add(2, 3)).thenReturn(5);
assertEquals(5, calc.add(2, 3));
```
