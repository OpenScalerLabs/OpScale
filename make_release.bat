@echo off
setlocal EnableExtensions

title OpScale Release Builder
cd /d "%~dp0"

echo ==================================================
echo          OpScale Release Builder
echo ==================================================
echo.

:: ----------------------------------------------------
:: 1. Python prüfen
:: ----------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python wurde nicht gefunden.
    pause
    exit /b 1
)

echo [1/4] Python gefunden.

:: ----------------------------------------------------
:: 2. PyInstaller prüfen / installieren
:: ----------------------------------------------------
python -m PyInstaller --version >nul 2>&1

if errorlevel 1 (
    echo [2/4] PyInstaller nicht gefunden. Installiere...
    python -m pip install pyinstaller

    python -m PyInstaller --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo FEHLER: PyInstaller konnte nicht installiert werden.
        pause
        exit /b 1
    )
)

echo [2/4] PyInstaller ist bereit.

:: ----------------------------------------------------
:: 3. Alte Build-Dateien entfernen
:: ----------------------------------------------------
echo [3/4] Entferne alte Build-Dateien...

if exist build rd /s /q build
if exist dist rd /s /q dist
if exist __pycache__ rd /s /q __pycache__
del /q *.spec >nul 2>&1

echo Fertig.
echo.

:: ----------------------------------------------------
:: 4. EXE erstellen
:: ----------------------------------------------------
echo [4/4] Erstelle OpScale.exe...
echo.

python -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --onefile ^
    --console ^
    --name OpScale ^
    --icon "images\OpScale.ico" ^
    --collect-all onnxruntime ^
    --hidden-import onnxruntime ^
    --hidden-import onnxruntime.capi ^
    --hidden-import onnxruntime.capi.onnxruntime_pybind11_state ^
    main.py

if errorlevel 1 (
    echo.
    echo ==========================================
    echo FEHLER: PyInstaller Build fehlgeschlagen.
    echo ==========================================
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Build erfolgreich abgeschlossen!
echo ==========================================
echo.

echo Die EXE befindet sich unter:
echo.
echo %CD%\dist\OpScale.exe
echo.

if exist "dist\OpScale.exe" explorer "dist"

pause