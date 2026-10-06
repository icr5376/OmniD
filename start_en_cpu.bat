@echo off
cd /d "%~dp0"
echo Starting OmniDictate (English version / CPU mode)...
echo VRAM usage: 0GB - Transcription will be slightly slower.
echo.
set PYTHONUTF8=1
venv\Scripts\python.exe dictation_en.py --cpu
echo.
echo Application closed.
pause
