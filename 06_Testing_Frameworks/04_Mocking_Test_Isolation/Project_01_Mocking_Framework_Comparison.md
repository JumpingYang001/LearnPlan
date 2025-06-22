# Project: Mocking Framework Comparison

## Description
Build a test suite using different mocking frameworks. Implement the same tests across multiple frameworks and document the differences in approach and syntax.

## Example (Python, Java, C#)

### Python (unittest.mock)
```python
from unittest.mock import Mock
calc = Mock()
calc.add.return_value = 5
assert calc.add(2, 3) == 5
```

### Java (Mockito)
```java
Calculator calc = mock(Calculator.class);
when(calc.add(2, 3)).thenReturn(5);
assertEquals(5, calc.add(2, 3));
```

### C# (Moq)
```csharp
var mock = new Mock<ICalculator>();
mock.Setup(m => m.Add(2, 3)).Returns(5);
Assert.Equal(5, mock.Object.Add(2, 3));
```
