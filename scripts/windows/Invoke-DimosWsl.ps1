[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$LinuxCommand,
    [string]$Distro = "Ubuntu",
    [switch]$UseRepoMount,
    [switch]$StopExistingCoordinator
)

$ErrorActionPreference = "Stop"

function Get-WslRepoDir {
    param([switch]$UseRepoMount)

    if ($UseRepoMount) {
        $drive = $PWD.Path.Substring(0, 1).ToLowerInvariant()
        $rest = $PWD.Path.Substring(2).Replace('\', '/')
        return "/mnt/$drive$rest"
    }

    return '$HOME/dimos'
}

$repoDir = Get-WslRepoDir -UseRepoMount:$UseRepoMount
$commandPrefix = ""
if ($StopExistingCoordinator) {
    $commandPrefix = ".venv/bin/python -m dimos.robot.cli.dimos stop >/dev/null 2>&1 || true; "
}
$command = "cd $repoDir && $commandPrefix$LinuxCommand"

Write-Host "[dimos-wsl] distro=$Distro"
Write-Host "[dimos-wsl] repo=$repoDir"
Write-Host "[dimos-wsl] cmd=$LinuxCommand"
if ($StopExistingCoordinator) { Write-Host "[dimos-wsl] preflight=dimos stop" }

wsl.exe -d $Distro -- bash -lc $command
exit $LASTEXITCODE
