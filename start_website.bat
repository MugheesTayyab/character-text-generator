@echo off
echo Starting MatteMind Web Application...
echo.
echo Please wait while the model loads into memory. This may take a few seconds.
echo Once loaded, your browser will open automatically.
echo.

:: Start the Flask app in the background
start "MatteMind Backend" cmd /c "python app.py"

:: Wait a few seconds for the Flask server to initialize
timeout /t 3 /nobreak > nul

:: Open the default browser to the local server
start http://127.0.0.1:5000/

echo Server is running! Keep the backend window open while using the app.
pause
