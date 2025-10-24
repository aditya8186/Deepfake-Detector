@echo off
echo Starting Deepfake Detector Flask Application...
echo.

REM Check if virtual environment exists
if not exist "..\venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please make sure you have created a virtual environment in the parent directory.
    echo Run: python -m venv ..\venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call "..\venv\Scripts\activate.bat"

REM Check if we're in the right directory
if not exist "app\app.py" (
    echo Error: app\app.py not found!
    echo Please run this script from the deepfake_detector directory.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Installing/updating dependencies...
pip install -r requirements.txt

REM Start the Flask application
echo.
echo Starting Flask application...
echo The app will be available at: http://localhost:5000
echo Press Ctrl+C to stop the application
echo.
cd app
python app.py

pause

