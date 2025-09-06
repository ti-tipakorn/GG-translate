@echo off
setlocal enableextensions

REM --- Paths (everything in C:\keys) ---
set BASE=C:\keys
set VENV=%BASE%\venv
set LOGDIR=%BASE%\logs
set SCRIPT=%BASE%\daily_english.py

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM --- Stamp & basic diagnostics ---
echo ==== START %date% %time% ====>> "%LOGDIR%\bot.out.log"
set PYTHONUTF8=1
echo USERNAME=%USERNAME%>> "%LOGDIR%\bot.out.log"
echo GOOGLE_APPLICATION_CREDENTIALS=%GOOGLE_APPLICATION_CREDENTIALS%>> "%LOGDIR%\bot.out.log"

REM --- Activate venv and run the bot with full path python ---
call "%VENV%\Scripts\activate.bat"
"%VENV%\Scripts\python.exe" -u "%SCRIPT%"  >> "%LOGDIR%\bot.out.log"  2>> "%LOGDIR%\bot.err.log"

echo ==== END   %date% %time% ====>> "%LOGDIR%\bot.out.log"
endlocal