# Project: Testing Asynchronous Code

## Description
Develop a system with asynchronous operations. Use mocking for callbacks, promises, and async/await to create deterministic tests for non-deterministic behavior.

## Example (JavaScript with Jest)
```javascript
test('async fetch', async () => {
  const fetchData = jest.fn().mockResolvedValue('data');
  await expect(fetchData()).resolves.toBe('data');
});
```
