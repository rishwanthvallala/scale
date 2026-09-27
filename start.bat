@echo off
rem Serves the site locally and opens it in your browser. Close this window to stop.
cd /d "%~dp0"
set PORT=8080
start "" "http://localhost:%PORT%/"
where py >nul 2>nul && (py -m http.server %PORT%) || (python -m http.server %PORT%)
