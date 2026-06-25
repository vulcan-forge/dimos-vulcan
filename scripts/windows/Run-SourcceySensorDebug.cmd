@echo off
setlocal
set SCRIPT_DIR=%~dp0
if /I not "%DIMOS_DISABLE_WINDOWS_PREVIEW%"=="1" (
  powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Start-SourcceyWindowsPreview.ps1"
)
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Invoke-DimosWsl.ps1" -LinuxCommand ".venv/bin/python -m dimos.robot.cli.dimos --viewer none --robot-ip ${DIMOS_ROBOT_IP:-192.168.1.237} run sourccey-sensor-debug-view"
endlocal
