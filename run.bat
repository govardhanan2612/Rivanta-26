@echo off
REM Start the app. Double-click this, or run it from a terminal.
cd /d "%~dp0"
echo Starting incident triage on http://localhost:8501
echo Close this window to stop the app.
python -m streamlit run app.py --server.port 8501
pause
