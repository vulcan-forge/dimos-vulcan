@echo off
setlocal
set SCRIPT_DIR=%~dp0
py -3 "%SCRIPT_DIR%watch_sourccey_spatial_preview.py" %*
endlocal
