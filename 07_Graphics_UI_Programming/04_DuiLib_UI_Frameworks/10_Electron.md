# Electron for Desktop Apps

## Topics
- Electron framework concepts
- Web technologies for desktop apps
- IPC and native integration
- Comparison with native UI frameworks

### Example: Electron Main Process (JavaScript)
```js
const { app, BrowserWindow } = require('electron');
app.whenReady().then(() => {
  const win = new BrowserWindow({ width: 800, height: 600 });
  win.loadFile('index.html');
});
```
