# Mock Frameworks for .NET

## Description
Popular .NET mocking frameworks include Moq and NSubstitute, which help create test doubles and verify interactions.

## Example (C#)
```csharp
var mock = new Mock<ICalculator>();
mock.Setup(m => m.Add(2, 3)).Returns(5);
Assert.Equal(5, mock.Object.Add(2, 3));
```
