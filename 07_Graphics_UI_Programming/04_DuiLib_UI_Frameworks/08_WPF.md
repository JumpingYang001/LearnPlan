# Windows Presentation Foundation (WPF)

## Topics
- XAML and WPF architecture
- Data binding and MVVM
- Styling and templating
- Comparison with DirectUI approaches

### Example: Simple WPF XAML
```xml
<Window x:Class="WpfApp.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="WPF Window" Height="300" Width="400">
    <Grid>
        <Button Content="Click Me" HorizontalAlignment="Center" VerticalAlignment="Center"/>
    </Grid>
</Window>
```
