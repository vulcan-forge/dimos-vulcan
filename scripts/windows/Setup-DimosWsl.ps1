[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu",
    [string]$WslRepoDir = '$HOME/dimos',
    [string]$Extras = "base,cuda,sim",
    [string]$SourceRepoPath = $PWD.Path,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Require-Command "wsl.exe"
Require-Command "git.exe"

function Get-OptionalFeatureState {
    param([string]$FeatureName)
    try {
        return (Get-WindowsOptionalFeature -Online -FeatureName $FeatureName -ErrorAction Stop).State
    } catch {
        return $null
    }
}

function Test-FirmwareVirtualizationEnabled {
    try {
        $processors = Get-CimInstance Win32_Processor -ErrorAction Stop
        foreach ($processor in $processors) {
            if (-not $processor.VirtualizationFirmwareEnabled) {
                return $false
            }
        }
        return $true
    } catch {
        return $null
    }
}

$wslStatus = & wsl.exe --status 2>$null | Out-String
if (-not $wslStatus) {
    throw "WSL is not available. Install it first with: wsl --install -d Ubuntu"
}

$wslFeature = Get-OptionalFeatureState -FeatureName "Microsoft-Windows-Subsystem-Linux"
$vmPlatformFeature = Get-OptionalFeatureState -FeatureName "VirtualMachinePlatform"
$firmwareVirtualization = Test-FirmwareVirtualizationEnabled

if ($wslFeature -eq "Disabled") {
    throw @"
Windows Subsystem for Linux is not enabled yet.

Run this in an elevated PowerShell window:
  wsl --install --no-distribution

Then reboot Windows and run this setup again.
"@
}

if ($vmPlatformFeature -eq "Disabled") {
    throw @"
Virtual Machine Platform is not enabled yet.

Run this in an elevated PowerShell window:
  wsl --install --no-distribution

Then reboot Windows and run this setup again.
"@
}

if ($firmwareVirtualization -eq $false) {
    throw @"
CPU virtualization appears to be disabled in firmware/BIOS.

Please reboot into BIOS/UEFI and enable the virtualization setting first.
Common names include:
  Intel VT-x
  Intel Virtualization Technology
  SVM Mode
  AMD-V

After enabling it, boot back into Windows and run this setup again.
"@
}

if ($null -eq $wslFeature -or $null -eq $vmPlatformFeature) {
    Write-Host "[dimos-wsl] optional feature state could not be queried without elevation; continuing because 'wsl --status' succeeded."
}

$distros = & wsl.exe -l -q | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($distros -notcontains $Distro) {
    throw "WSL distro '$Distro' is not installed. Install it first with: wsl --install -d $Distro"
}

if ($SkipInstall) {
    Write-Host "[dimos-wsl] WSL looks available. Skipping install step."
    exit 0
}

function Convert-ToWslPath {
    param([string]$WindowsPath)
    $resolved = (Resolve-Path $WindowsPath).Path
    $drive = $resolved.Substring(0, 1).ToLowerInvariant()
    $rest = $resolved.Substring(2).Replace('\', '/')
    return "/mnt/$drive$rest"
}

$scriptPath = "$(Convert-ToWslPath -WindowsPath $PSScriptRoot)/../wsl/install_dimos_wsl.sh"
$sourceRepoWslPath = Convert-ToWslPath -WindowsPath $SourceRepoPath
$linuxCommand = "export DIMOS_WSL_EXTRAS='$Extras'; bash $scriptPath '$WslRepoDir' '$sourceRepoWslPath'"

Write-Host "[dimos-wsl] bootstrapping DimOS in WSL distro '$Distro'"
wsl.exe -d $Distro -- bash -lc $linuxCommand
exit $LASTEXITCODE
