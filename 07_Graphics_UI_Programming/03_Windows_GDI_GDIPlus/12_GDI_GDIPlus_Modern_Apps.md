# GDI/GDI+ in Modern Windows Applications

## Topics
- GDI/GDI+ in Windows Forms
- GDI/GDI+ in WPF applications
- GDI/GDI+ in Win32 applications
- GDI/GDI+ in different application frameworks

### Example: Using GDI+ in Windows Forms (C#)
```csharp
protected override void OnPaint(PaintEventArgs e)
{
    e.Graphics.DrawEllipse(Pens.Blue, 10, 10, 100, 50);
}
```
