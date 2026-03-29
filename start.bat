@echo off
cd /d "%~dp0server"
set PYTHONPATH=%~dp0server
..\python\python.exe main.py
pause