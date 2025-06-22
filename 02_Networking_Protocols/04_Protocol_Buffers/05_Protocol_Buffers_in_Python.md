# Protocol Buffers in Python

## Using Protocol Buffers in Python
Generate Python code using protoc and use the generated classes.

## Compile Command
```sh
protoc --python_out=. person.proto
```

## Python Example
```python
import person_pb2

person = person_pb2.Person()
person.name = "Carol"
person.id = 789

serialized = person.SerializeToString()

person2 = person_pb2.Person()
person2.ParseFromString(serialized)
```
