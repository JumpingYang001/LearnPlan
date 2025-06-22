# Protocol Buffers in Java/Kotlin

## Using Protocol Buffers in Java
Generate Java code using protoc and use the generated classes.

## Compile Command
```sh
protoc --java_out=. person.proto
```

## Java Example
```java
Person person = Person.newBuilder()
    .setName("Dave")
    .setId(101)
    .build();

byte[] data = person.toByteArray();

Person parsed = Person.parseFrom(data);
```
