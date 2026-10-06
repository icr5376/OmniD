@echo off
cd /d "%~dp0"
echo Starting OmniDictate (Japanese + AI Correction)...
echo Ollama high-precision mode: kanji correction enabled.
echo.
set PYTHONUTF8=1
REM Whisper -> RTX 3060 (the RTX 4070 Ti is reserved for Ollama + display).
set OMNIDICTATE_GPU=GPU-11198d1e-3bf8-ab86-6142-14a0e9cbcb81
venv\Scripts\python.exe dictation.py --ai
echo.
echo Application closed.
pause
