@echo off
echo Installing dependencies...
python\python.exe -m pip install -r server\requirements-common.txt -r server\requirements-cuda.txt
echo Done! Run start.bat to launch.
pause