@echo off
echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing modules...
python -m pip install opencv-python cvzone numpy

echo Verifying installation...
python -c "import cv2, cvzone, numpy; print('All modules installed successfully!')"

echo Done!
pause