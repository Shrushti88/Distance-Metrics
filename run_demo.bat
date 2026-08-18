@echo off
title Similarity Measurement Using Distance Metrics - PBL Project
echo ===============================================================================
echo   Similarity Measurement Using Distance Metrics: Euclidean, Manhattan, Cosine
echo   Pattern Recognition Project-Based Learning (PBL)
echo ===============================================================================
echo.
echo [1/2] Running Benchmark Pipeline and generating all publication charts...
python main.py
echo.
echo [2/2] Launching Interactive Web Dashboard at http://127.0.0.1:5000...
echo Open your web browser and navigate to: http://127.0.0.1:5000
echo Press Ctrl+C in this terminal window to stop the server.
echo.
python web_app/app.py
pause
