$ErrorActionPreference = 'SilentlyContinue'

# Stop old Streamlit instances for this app
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python|pythonw' -and $_.CommandLine -match 'streamlit run app.py'
}
if ($procs) {
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Start-Sleep -Milliseconds 800
}

# Free default port if still occupied
$pids = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($pids) {
    $pids | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Milliseconds 500
}

# Use venv Python if available, otherwise fall back to system Python
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$sysPy  = 'C:/Users/kalan/AppData/Local/Python/pythoncore-3.14-64/python.exe'
$python = if (Test-Path $venvPy) { $venvPy } else { $sysPy }

Write-Host "Using Python: $python"
Write-Host 'Starting Streamlit on http://localhost:8501 ...'

# Suppress Python warnings; run with no file watcher to prevent VS Code crashes
$env:PYTHONWARNINGS = 'ignore'
$env:WM_DEMO_QUICK  = '1'

& $python -W ignore -m streamlit run app.py `
    --server.port 8501 `
    --server.headless true `
    --server.fileWatcherType none `
    --server.runOnSave false `
    --logger.level error
