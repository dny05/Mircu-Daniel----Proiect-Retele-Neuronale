@echo off
echo ============================================
echo  Suspension Setup Evaluator
echo ============================================
echo.

REM Verifică dacă există venv
if not exist "venv" (
    echo Creez mediul virtual...
    python -m venv venv
)

REM Activează venv
echo Activez mediul virtual...
call venv\Scripts\activate.bat

REM Verifică dacă sunt instalate pachetele
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Instalez dependințele...
    pip install -r requirements.txt
)

REM Pornește aplicația
echo.
echo ============================================
echo  Pornesc aplicația...
echo  Accesează: http://localhost:8501
echo ============================================
echo.
streamlit run app.py

pause
