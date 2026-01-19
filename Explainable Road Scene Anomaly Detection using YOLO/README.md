# Explainable Road Scene Anomaly Detection System

## Academic Demonstration (30% Implementation)

This project demonstrates a simplified road scene anomaly detection system using YOLO for object detection with basic explainability features.

### Implemented Modules

1. **Data Collection Module** - Load traffic images or videos from local folder
2. **Preprocessing Module** - Resize input frames to YOLO input size and normalize
3. **Detection Module** - Use pre-trained YOLO model to detect vehicles and pedestrians
4. **Basic Explainability Module** - Draw bounding boxes and confidence scores

### Requirements

- Python 3.8+
- ultralytics (YOLO)
- OpenCV
- PyTorch
- NumPy

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The system will automatically download YOLOv8n model on first run.

### Usage

#### Process an Image
```bash
python detection.py --input path/to/your/image.jpg
```

#### Process a Video
```bash
python detection.py --input path/to/your/video.mp4
```

#### Use Custom YOLO Model
```bash
python detection.py --input path/to/input --model path/to/custom/model.pt
```

### Detected Classes

The system detects the following road scene objects:
- **Pedestrians** (Green boxes)
- **Bicycles** (Cyan boxes)
- **Cars** (Blue boxes)
- **Motorcycles** (Orange boxes)
- **Buses** (Magenta boxes)
- **Trucks** (Red boxes)

### Controls

- **For Images**: Press any key to close the window
- **For Videos**: Press 'q' to quit during playback

### Sample Test Data

You can test the system with:
- Any traffic images (JPG, PNG, BMP formats)
- Any traffic videos (MP4, AVI, MOV formats)
- Sample data from traffic datasets or dashcam footage

### Academic Note

This is a demonstration implementation focusing on core functionality. The full system would include:
- Advanced anomaly classification algorithms
- CNN comparison models (VGG16, ResNet, DenseNet)
- Training and evaluation metrics
- Advanced XAI techniques (Grad-CAM, heatmaps)
- Real-time performance optimization