@echo off
setlocal
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Invoke-DimosWsl.ps1" -LinuxCommand ".venv/bin/python -m dimos.robot.cli.dimos stop"
endlocal
