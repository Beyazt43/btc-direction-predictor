# btcpred watchdog
#
# Brings the stack back when it dies without anyone watching. Two real failure
# modes have cost prediction slots on this host:
#
#   1. Windows Update restarts the machine; Docker Desktop is a per-user GUI app
#      and does not return until someone logs in.
#   2. A Microsoft Store update to WSL tears down the WSL VM, taking the Docker
#      daemon and every container with it. `restart: unless-stopped` cannot help
#      because the daemon itself is gone.
#
# Both leave the stack down indefinitely. This runs on a timer and repairs it.
#
# It also re-checks the health endpoint FROM THE HOST: after a WSL teardown the
# container healthchecks pass (they run in-container) while the host port proxy
# stays broken, so `docker compose ps` looks fine and the dashboard is dead.
#
# Create .watchdog-disabled in the repo root to stop it interfering during
# maintenance.
#
# Register it as a task that runs as you, every five minutes, with no password
# stored. Two triggers are needed: an at-logon trigger's repetition only starts
# at a logon, so on its own the task sits idle until the next one. Tasks also
# default to not running on battery, which silently disables it on a laptop.
#
#   $me  = "$env:COMPUTERNAME\$env:USERNAME"
#   $act = New-ScheduledTaskAction -Execute powershell.exe -Argument `
#          "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PWD\scripts\watchdog.ps1`""
#   $rep = (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)).Repetition
#   $t1  = New-ScheduledTaskTrigger -AtLogOn -User $me; $t1.Repetition = $rep
#   $t2  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
#   $set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
#          -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
#   Register-ScheduledTask -TaskName btcpred-watchdog -Action $act -Trigger @($t1, $t2) -Settings $set `
#          -Principal (New-ScheduledTaskPrincipal -UserId $me -LogonType Interactive -RunLevel Limited)
#
# Run that from the repo root. Check it with Get-ScheduledTaskInfo: an empty
# NextRunTime means the repeating trigger did not take.

$ErrorActionPreference = 'Continue'

# Paths are derived rather than hardcoded: the repo is this script's parent
# directory, and Docker Desktop is looked for in its per-user install location
# first, then the system-wide one.
$Repo    = Split-Path -Parent $PSScriptRoot
$DDRoot  = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop'
if (-not (Test-Path $DDRoot)) { $DDRoot = Join-Path $env:ProgramFiles 'Docker\Docker' }
$Docker  = Join-Path $DDRoot 'resources\bin\docker.exe'
$DDExe   = Join-Path $DDRoot 'Docker Desktop.exe'
$Health  = 'http://localhost:8000/health'
$LogDir  = Join-Path $env:LOCALAPPDATA 'btcpred-watchdog'
$Log     = Join-Path $LogDir 'watchdog.log'
$Disable = Join-Path $Repo '.watchdog-disabled'

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $Log -Value $line -Encoding utf8
}

# Keep the log from growing without bound.
if ((Test-Path $Log) -and ((Get-Item $Log).Length -gt 1MB)) {
    $keep = Get-Content $Log -Tail 500
    Set-Content -Path $Log -Value $keep -Encoding utf8
}

if (Test-Path $Disable) { exit 0 }

function Test-Daemon {
    try {
        & $Docker version --format '{{.Server.Version}}' | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

# --- 1. daemon ------------------------------------------------------------
if (-not (Test-Daemon)) {
    Write-Log 'daemon unreachable'

    if (-not (Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue)) {
        Write-Log 'starting Docker Desktop'
        try { Start-Process $DDExe } catch { Write-Log "could not start Docker Desktop: $_" }
    } else {
        # Processes alive but no daemon: the usual shape after a WSL teardown.
        Write-Log 'Docker Desktop running but daemon dead; restarting it'
        try {
            & $Docker desktop restart | Out-Null
        } catch { Write-Log "desktop restart failed: $_" }
    }

    $up = $false
    for ($i = 0; $i -lt 36; $i++) {
        Start-Sleep -Seconds 5
        if (Test-Daemon) { $up = $true; break }
    }
    if (-not $up) { Write-Log 'daemon still down after 180s; will retry next cycle'; exit 1 }
    Write-Log 'daemon is up'
}

# --- 2. stack -------------------------------------------------------------
Push-Location $Repo
try {
    $running = & $Docker compose ps --status running --format '{{.Service}}'
    $names = @($running | Where-Object { $_ })
    $needed = @('db', 'api', 'scheduler')
    $missing = @($needed | Where-Object { $names -notcontains $_ })

    if ($missing.Count -gt 0) {
        Write-Log "services down: $($missing -join ', '); running compose up -d"
        & $Docker compose up -d | Out-Null
        Start-Sleep -Seconds 20
    }
} catch {
    Write-Log "compose check failed: $_"
} finally {
    Pop-Location
}

# --- 3. host-side health (catches the broken port proxy) -------------------
$healthy = $false
try {
    $r = Invoke-WebRequest -Uri $Health -TimeoutSec 10 -UseBasicParsing
    if ($r.StatusCode -eq 200) { $healthy = $true }
} catch {
    $healthy = $false
}

if (-not $healthy) {
    Write-Log 'health endpoint unreachable from host; restarting api'
    Push-Location $Repo
    try {
        & $Docker compose restart api | Out-Null
        Start-Sleep -Seconds 15
        try {
            $r2 = Invoke-WebRequest -Uri $Health -TimeoutSec 10 -UseBasicParsing
            if ($r2.StatusCode -eq 200) { Write-Log 'health restored' }
            else { Write-Log "health still bad (status $($r2.StatusCode))" }
        } catch {
            Write-Log 'health still unreachable after api restart'
        }
    } finally { Pop-Location }
}
