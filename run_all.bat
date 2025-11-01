@echo off
REM Windows batch script to run all experiments

echo ================================================================================
echo RUNNING ALL EXPERIMENTS
echo ================================================================================
echo.

echo [1/4] Training ResNet18 Baseline...
py train_resnet_baseline.py
if %errorlevel% neq 0 (
    echo ERROR: Baseline training failed
    pause
    exit /b %errorlevel%
)
echo.

echo [2/4] Training ResNet18 + Forward Loss Correction...
py train_forward_correction.py
if %errorlevel% neq 0 (
    echo ERROR: Forward correction training failed
    pause
    exit /b %errorlevel%
)
echo.

echo [3/4] Training ResNet18 + Backward Loss Correction...
py train_backward_correction.py
if %errorlevel% neq 0 (
    echo ERROR: Backward correction training failed
    pause
    exit /b %errorlevel%
)
echo.

echo [4/4] Training ResNet18 + Co-Teaching...
py train_coteaching.py
if %errorlevel% neq 0 (
    echo ERROR: Co-Teaching training failed
    pause
    exit /b %errorlevel%
)
echo.

echo ================================================================================
echo ALL EXPERIMENTS COMPLETED
echo ================================================================================
echo.
echo Output files:
echo   - resnet_baseline.pth ^& resnet_baseline_results.csv
echo   - resnet_forward.pth ^& resnet_forward_results.csv
echo   - resnet_backward.pth ^& resnet_backward_results.csv
echo   - resnet_coteaching.pth ^& resnet_coteaching_results.csv
echo.
pause

