@echo off
setlocal
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Invoke-DimosWsl.ps1" -StopExistingCoordinator -LinuxCommand ".venv/bin/python -m dimos.robot.cli.dimos --viewer none --robot-ip ${DIMOS_ROBOT_IP:-192.168.1.237} run sourccey-basic"
endlocal
