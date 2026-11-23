$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backendProcess = $null
$frontendProcess = $null

function Test-Port($Port) {
    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        return $null -ne $connection
    } catch {
        return $false
    }
}

Set-Location $root

function Find-Python {
    foreach ($command in @(
        @{ File = "py"; Arguments = @("-3.12") },
        @{ File = "py"; Arguments = @("-3.13") },
        @{ File = "py"; Arguments = @("-3.11") },
        @{ File = "py"; Arguments = @("-3.10") }
    )) {
        try {
            $probeArgs = $command.Arguments + @("-c", "import sys; print(sys.version_info[:2])")
            $probe = & $command.File @probeArgs 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe -match "^\((3,\s*(10|11|12|13))\)") {
                return $command
            }
        } catch {}
    }
    foreach ($pythonFile in @("python", "python3")) {
        try {
            $probe = & $pythonFile "-c" "import sys; print(sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe -match "^\((3,\s*(10|11|12|13))\)") {
                return @{ File = $pythonFile; Arguments = @() }
            }
        } catch {}
    }
    return $null
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $pythonCommand = Find-Python
    if ($null -eq $pythonCommand) {
        throw "Python 3.10 or newer was not found. Install Python and enable Add Python to PATH, then rerun start.cmd."
    }
    Write-Host "Creating Python environment..."
    $createArgs = $pythonCommand.Arguments + @("-m", "venv", (Join-Path $root ".venv"))
    & $pythonCommand.File @createArgs
    if ($LASTEXITCODE -ne 0) { throw "Python virtual environment creation failed." }
}
else {
    $venvVersion = & $venvPython "-c" "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($LASTEXITCODE -ne 0 -or $venvVersion -notmatch "^3\.(10|11|12|13)$") {
        throw "The existing .venv uses unsupported Python $venvVersion. Delete .venv and rerun start.cmd."
    }
}

Write-Host "Checking backend dependencies..."
& $venvPython "-m" "pip" "install" "-r" (Join-Path $root "backend\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }

if (-not (Test-Port 5173)) {
    if (-not (Test-Path "frontend\node_modules")) {
        Write-Host "Installing frontend dependencies..."
        & npm.cmd --prefix frontend install
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
    }
    $frontendProcess = Start-Process -FilePath "npm.cmd" `
        -ArgumentList "--prefix", "frontend", "run", "dev" `
        -WorkingDirectory $root `
        -PassThru
}

if (-not (Test-Port 8000)) {
    $venvPython = Join-Path $root ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Python environment not found at $venvPython. Create it with: py -3.10 -m venv .venv"
    }
    $backendProcess = Start-Process -FilePath $venvPython `
        -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $root `
        -PassThru
}

$backendReady = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if (Test-Port 8000) {
        $backendReady = $true
        break
    }
    if ($null -ne $backendProcess -and $backendProcess.HasExited) {
        break
    }
    Start-Sleep -Milliseconds 250
}

if (-not $backendReady) {
    throw "Backend did not start on port 8000. Check Python dependencies with: .\.venv\Scripts\python.exe -m pip check"
}

$frontendReady = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if (Test-Port 5173) {
        $frontendReady = $true
        break
    }
    Start-Sleep -Milliseconds 250
}

if (-not $frontendReady) {
    throw "Frontend did not start on port 5173."
}

Write-Host ""
Write-Host "FSLT is running:"
Write-Host "  Web:     http://localhost:5173"
Write-Host "  Backend: http://127.0.0.1:8000"
Write-Host ""
Write-Host "Press Ctrl+C to stop processes launched by this script."

Start-Process "http://localhost:5173"

try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    foreach ($process in @($frontendProcess, $backendProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
        }
    }
}
