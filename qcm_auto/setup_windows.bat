@echo off
echo ============================================
echo   SETUP QCM AUTO - Environnement Anaconda
echo ============================================
echo.

echo [1/4] Creation de l'environnement conda qcm_auto...
call conda create -n qcm_auto python=3.10 -y

echo.
echo [2/4] Activation de l'environnement...
call conda activate qcm_auto

echo.
echo [3/4] Installation de ffmpeg...
call conda install -c conda-forge ffmpeg -y

echo.
echo [4/4] Installation des dependances Python...
pip install -r requirements.txt

echo.
echo ============================================
echo   INSTALLATION TERMINEE !
echo ============================================
echo.
echo Pour lancer le portail professeur :
echo   conda activate qcm_auto
echo   python interface/app.py
echo.
echo Puis ouvrir dans le navigateur : http://localhost:5000
echo.
pause
