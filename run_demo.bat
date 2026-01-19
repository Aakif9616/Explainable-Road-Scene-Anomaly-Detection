@echo off
echo ============================================================
echo Road Scene Anomaly Detection - Quick Demo
echo ============================================================
echo.

echo 1. Testing setup...
python test_setup.py
echo.

echo 2. Checking available images...
python quick_test.py
echo.

echo 3. Running detection on your images...
echo.

echo Testing Morning Road Image:
python detection.py --input "my_images/morning road image.jpg"
echo.

echo Testing Night Road Image:
python detection.py --input "my_images/night road image.jpg"
echo.

echo Testing Regular Road Image:
python detection.py --input "my_images/road image.jpg"
echo.

echo ============================================================
echo Demo completed! Press any key to exit...
pause