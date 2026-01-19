"""
Test script to verify the road scene detection system setup
"""

import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    required_packages = [
        'cv2',
        'numpy', 
        'ultralytics',
        'torch',
        'PIL'
    ]
    
    print("Testing package imports...")
    print("-" * 40)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - {e}")
            failed_imports.append(package)
    
    print("-" * 40)
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} package(s) failed to import:")
        for package in failed_imports:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_yolo_model():
    """Test YOLO model loading"""
    print("\nTesting YOLO model loading...")
    print("-" * 40)
    
    try:
        from ultralytics import YOLO
        
        # This will download YOLOv8n if not present
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully!")
        
        # Test model info
        print(f"Model classes: {len(model.names)}")
        print("Sample classes:", list(model.names.values())[:10])
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV functionality...")
    print("-" * 40)
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = (255, 0, 0)  # Blue image
        
        # Test basic OpenCV operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(test_image, (50, 50))
        
        print("✅ OpenCV basic operations working!")
        print(f"OpenCV version: {cv2.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Road Scene Detection System - Setup Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test OpenCV
    if test_opencv():
        tests_passed += 1
    
    # Test YOLO model
    if test_yolo_model():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 Setup verification completed successfully!")
        print("\nYou can now run the detection system:")
        print("python detection.py --input path/to/your/image_or_video")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("Make sure to install all requirements:")
        print("pip install -r requirements.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    main()