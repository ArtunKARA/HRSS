@echo off
setlocal
set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%run_bv_full_suite.py"
exit /b %ERRORLEVEL%
