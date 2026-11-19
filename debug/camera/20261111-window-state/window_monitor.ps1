param([string]$OutputPath, [string]$StopPath, [int]$TargetProcessId)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class UEWindowMonitor {
 [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
 [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr handle);
 [DllImport("user32.dll")] public static extern bool IsZoomed(IntPtr handle);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr handle, out uint processId);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr handle, out Rect rect);
}
'@
$timer = [Diagnostics.Stopwatch]::StartNew()
while ($timer.Elapsed.TotalSeconds -lt 180 -and -not (Test-Path -LiteralPath $StopPath)) {
    $editor = Get-Process -Id $TargetProcessId
    $foregroundProcessId = [uint32]0
    [void][UEWindowMonitor]::GetWindowThreadProcessId([UEWindowMonitor]::GetForegroundWindow(), [ref]$foregroundProcessId)
    $rect = New-Object UEWindowMonitor+Rect
    [void][UEWindowMonitor]::GetWindowRect($editor.MainWindowHandle, [ref]$rect)
    $sample = [ordered]@{
        time = [DateTimeOffset]::Now.ToString('o')
        process_id = $editor.Id
        minimized = [UEWindowMonitor]::IsIconic($editor.MainWindowHandle)
        maximized = [UEWindowMonitor]::IsZoomed($editor.MainWindowHandle)
        foreground = ($editor.Id -eq $foregroundProcessId)
        width = $rect.Right - $rect.Left
        height = $rect.Bottom - $rect.Top
    }
    [IO.File]::AppendAllText($OutputPath, ($sample | ConvertTo-Json -Compress) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    Start-Sleep -Milliseconds 1000
}
