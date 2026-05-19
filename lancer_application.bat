@echo off
echo ============================================
echo   QCM AUTO - Lancement du portail
echo ============================================
echo.
call conda activate qcm_auto
echo Portail accessible sur : http://localhost:5000
echo Appuyez sur CTRL+C pour arreter le serveur
echo.
cd /d "%~dp0"
python interface/app.py
pause
