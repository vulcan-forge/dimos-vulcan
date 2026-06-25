[CmdletBinding()]
param(
    [string]$RobotIp = "192.168.1.237",
    [int]$TileWidth = 480,
    [int]$MaxColumns = 2
)

$ErrorActionPreference = "Stop"

$scriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$dimosRepo = Split-Path -Parent (Split-Path -Parent $scriptsDir)
$slamRepo = Join-Path (Split-Path -Parent $dimosRepo) "sourccey-slam"

if (-not (Test-Path $slamRepo)) {
    throw "Could not find sibling sourccey-slam repo at '$slamRepo'."
}

$slamExe = Join-Path $slamRepo ".venv\Scripts\sourccey-slam.exe"
$slamPy = Join-Path $slamRepo ".venv\Scripts\python.exe"

if (Test-Path $slamExe) {
    $launchCommand = @(
        "Set-Location '$slamRepo'"
        "& '$slamExe' camera-preview-all --config config/depth_room_mapping.yaml --endpoint tcp://$RobotIp`:5560 --tile-width $TileWidth --max-columns $MaxColumns"
    ) -join "; "
}
elseif (Test-Path $slamPy) {
    $launchCommand = @(
        "Set-Location '$slamRepo'"
        "& '$slamPy' -m sourccey_slam.app.cli camera-preview-all --config config/depth_room_mapping.yaml --endpoint tcp://$RobotIp`:5560 --tile-width $TileWidth --max-columns $MaxColumns"
    ) -join "; "
}
else {
    throw "Could not find sourccey-slam virtualenv Python or CLI under '$slamRepo\.venv\Scripts'."
}

Write-Host "[dimos-preview] launching native Windows preview from $slamRepo"
Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $launchCommand
) | Out-Null
