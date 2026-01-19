# 🚀 Complete Setup Guide: Road Scene Anomaly Detection in VS Code

## 📋 Prerequisites
- Python 3.8+ installed
- VS Code installed
- Internet connection (for downloading packages)

---

## 🔧 Step 1: VS Code Setup

### 1.1 Install VS Code Extensions
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Install these extensions:
   - **Python** (by Microsoft)
   - **Python Debugger** (by Microsoft)
   - **Pylance** (by Microsoft)

### 1.2 Create Project Folder
1. Create a new folder: `Road_Scene_Detection`
2. Open VS Code
3. File → Open Folder → Select your project folder

---

## 📁 Step 2: Project Files Setup

### 2.1 Create All Required Files
Copy these files to your project folder:

**File 1: `detection.py`** (Main detection script)
**File 2: `requirements.txt`** (Dependencies)
**File 3: `test_setup.py`** (Setup verification)
**File 4: `README.md`** (Documentation)

### 2.2 Create Folders
```
Road_Scene_Detection/
├── detection.py
├── requirements.txt
├── test_setup.py
├── README.md
├── my_images/          # Create this folder
└── test_data/          # Create this folder
```

---

## ⚙️ Step 3: Environment Setup in VS Code

### 3.1 Open Terminal in VS Code
- Press `Ctrl + `` (backtick) to open terminal
- Or go to Terminal → New Terminal

### 3.2 Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected Output:**
```
Successfully installed ultralytics opencv-python torch numpy...
```

### 3.3 Verify Setup
```bash
python test_setup.py
```

**Expected Output:**
```
✅ All packages imported successfully!
✅ OpenCV basic operations working!
✅ YOLO model loaded successfully!
🎉 Setup verification completed successfully!
```

---

## 📸 Step 4: Add Test Images

### 4.1 Add Your Images
1. Save road scene images to `my_images/` folder
2. Supported formats: JPG, PNG, BMP
3. Good images: traffic scenes with cars, motorcycles, people

### 4.2 Check Available Images
```bash
python quick_test.py
```

---

## 🚗 Step 5: Run Detection

### 5.1 Basic Detection Command
```bash
python detection.py --input my_images/your_image.jpg
```

### 5.2 Test All Your Images
```bash
# Morning road scene
python detection.py --input "my_images/morning road image.jpg"

# Night road scene  
python detection.py --input "my_images/night road image.jpg"

# Regular road scene
python detection.py --input "my_images/road image.jpg"
```

### 5.3 Expected Output
```
✓ Detected 7 objects
  - person: 0.786
  - motorcycle: 0.661
  - car: 0.892
  - truck: 0.343
```

Plus a window showing your image with colored bounding boxes!

---

## 🎯 Step 6: VS Code Debugging Setup

### 6.1 Create Launch Configuration
1. Go to Run and Debug (Ctrl+Shift+D)
2. Click "create a launch.json file"
3. Select "Python File"

### 6.2 Edit launch.json
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Road Scene Detection",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/detection.py",
            "args": ["--input", "my_images/road image.jpg"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### 6.3 Run with Debugger
- Press F5 or click "Run and Debug"
- Set breakpoints by clicking left of line numbers
- Step through code with F10 (Step Over) and F11 (Step Into)

---

## 📊 Step 7: Understanding the Output

### 7.1 Console Output Explanation
```
Loading YOLO model from: yolov8n.pt     # Model loading
✓ Model loaded successfully              # Success confirmation
Processing image: my_images/road.jpg     # Input file
✓ Detected 7 objects                     # Number of detections
  - person: 0.786                        # Object type: confidence score
  - motorcycle: 0.661                    # Higher score = more confident
```

### 7.2 Visual Output
- **Green boxes**: Pedestrians (people)
- **Blue boxes**: Cars
- **Orange boxes**: Motorcycles  
- **Red boxes**: Trucks
- **Magenta boxes**: Buses
- **Cyan boxes**: Bicycles

### 7.3 Confidence Scores
- **0.9-1.0**: Very confident detection
- **0.7-0.9**: Good detection
- **0.5-0.7**: Moderate confidence
- **0.3-0.5**: Low confidence (still shown)

---

## 🔧 Step 8: Troubleshooting in VS Code

### 8.1 Common Issues

**Issue 1: "Could not load image"**
```bash
# Check file exists
ls my_images/
# Fix: Ensure correct file path and format
```

**Issue 2: Import errors**
```bash
# Reinstall packages
pip install --upgrade ultralytics opencv-python torch
```

**Issue 3: No display window**
```bash
# Check if running in headless environment
# Ensure you have display capability
```

### 8.2 VS Code Specific Fixes

**Python Interpreter Issues:**
1. Press Ctrl+Shift+P
2. Type "Python: Select Interpreter"
3. Choose the correct Python version

**Terminal Issues:**
1. Use VS Code integrated terminal
2. Ensure you're in the project directory
3. Check Python path with `which python` or `where python`

---

## 🎬 Step 9: Demo Preparation

### 9.1 Prepare Demo Images
1. Select 2-3 best road scene images
2. Test each one beforehand
3. Choose images with multiple object types

### 9.2 Demo Script
```bash
# Test 1: Morning traffic
python detection.py --input "my_images/morning road image.jpg"

# Test 2: Night scene
python detection.py --input "my_images/night road image.jpg"

# Test 3: Regular traffic
python detection.py --input "my_images/road image.jpg"
```

### 9.3 Explanation Points
1. **Show the code structure** in VS Code
2. **Explain YOLO architecture** (single-pass detection)
3. **Demonstrate real-time detection** on different images
4. **Highlight explainability** (colored boxes + confidence scores)
5. **Discuss practical applications** (traffic monitoring, safety)

---

## 📝 Step 10: Project Explanation for Mentor

### 10.1 Technical Overview
"This project implements explainable road scene anomaly detection using YOLOv8 deep learning model with four core modules: data collection, preprocessing, detection, and basic explainability."

### 10.2 Key Features
- **Real-time object detection** using state-of-the-art YOLO
- **Explainable AI** through visual bounding boxes and confidence scores
- **Multi-class detection** for road-relevant objects
- **Modular architecture** for easy extension

### 10.3 Live Demo Flow
1. Open VS Code with project
2. Show code structure and modules
3. Run detection on prepared images
4. Explain YOLO working principle
5. Highlight explainability features
6. Discuss real-world applications

---

## 🚀 Quick Start Commands

```bash
# 1. Setup verification
python test_setup.py

# 2. Check available images
python quick_test.py

# 3. Run detection
python detection.py --input "my_images/your_image.jpg"

# 4. Test all images
python detection.py --input "my_images/morning road image.jpg"
python detection.py --input "my_images/night road image.jpg"
python detection.py --input "my_images/road image.jpg"
```

---

## ✅ Success Checklist

- [ ] VS Code setup with Python extensions
- [ ] All project files created
- [ ] Dependencies installed successfully
- [ ] Setup verification passed
- [ ] Test images added to my_images folder
- [ ] Detection running successfully
- [ ] Visual output displaying correctly
- [ ] Demo script prepared
- [ ] Explanation points ready

**🎉 You're ready to demonstrate your explainable road scene anomaly detection system!**