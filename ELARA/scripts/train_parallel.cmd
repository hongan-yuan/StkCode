@echo off
setlocal

pushd "%~dp0\..\.."
if errorlevel 1 exit /b 1

where python >nul 2>nul
if errorlevel 1 goto use_py

python -m ELARA.parallel_runner train %*
set "ELARA_EXIT=%ERRORLEVEL%"
goto cleanup

:use_py
py -3 -m ELARA.parallel_runner train %*
set "ELARA_EXIT=%ERRORLEVEL%"

:cleanup
popd
exit /b %ELARA_EXIT%
