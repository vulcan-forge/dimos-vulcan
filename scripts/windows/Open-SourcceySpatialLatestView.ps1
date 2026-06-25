[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$path = "\\wsl$\Ubuntu\root\dimos\assets\output\memory\spatial_memory\reconstruction\latest_view.html"

if (-not (Test-Path $path)) {
    throw "Sourccey spatial viewer HTML not found at '$path'."
}

Start-Process explorer.exe -ArgumentList $path | Out-Null
