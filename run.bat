@echo off
REM Batch script to run preprocessing → training → deployment → Streamlit app

echo ================================
echo Running Preprocessing...
echo ================================
python -m src.preprocessing

echo ================================
echo Running Training (this will save models)...
echo ================================
python -m src.training

echo ================================
echo Testing Deployment...
echo ================================
python -m src.deployment
