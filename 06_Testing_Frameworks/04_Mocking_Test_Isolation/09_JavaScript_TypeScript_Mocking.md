# JavaScript/TypeScript Mocking

## Description
Jest and Sinon.js are popular libraries for mocking functions, modules, and timers in JavaScript/TypeScript.

## Example (JavaScript)
```javascript
const calc = { add: jest.fn().mockReturnValue(5) };
expect(calc.add(2, 3)).toBe(5);
```
