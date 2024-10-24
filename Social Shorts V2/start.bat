@echo off
REM Change directory to where the script is located
cd /d %~dp0

REM Activate the virtual environment
call .\venv\Scripts\activate

REM Run the Python script
python src\main.py

REM Optional: Pause to keep the window open after the script finishes
pause
