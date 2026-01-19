# 📸 How to Add Your Own Images for Detection

## 🗂️ Folder Structure
```
your_project/
├── detection.py          # Main detection script
├── my_images/           # 👈 PUT YOUR IMAGES HERE
│   ├── road_scene1.jpg
│   ├── traffic_photo.png
│   └── your_image.jpg
└── test_data/           # Sample images (optional)
```

## 📋 Step-by-Step Guide to Add Images

### Method 1: Copy Images Manually
1. **Save your road scene image** to your computer (right-click → Save As)
2. **Copy the image file** to the `my_images` folder
3. **Rename it** to something simple like `road_scene.jpg`

### Method 2: Using File Explorer
1. Open the project folder in File Explorer
2. Navigate to `my_images` folder
3. Drag and drop your image files here
4. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Method 3: Using Command Line
```bash
# Copy from Downloads folder (example)
copy "C:\Users\YourName\Downloads\your_image.jpg" my_images\road_scene.jpg
```

## 🚀 How to Run Detection on Your Images

### Single Image Detection
```bash
# Replace 'your_image.jpg' with your actual filename
python detection.py --input my_images/your_image.jpg
```

### Examples:
```bash
# Example 1: Road scene
python detection.py --input my_images/road_scene.jpg

# Example 2: Traffic photo
python detection.py --input my_images/traffic_photo.png

# Example 3: Night scene
python detection.py --input my_images/night_traffic.jpg
```

## 📝 Quick Test Commands

### Test with existing sample:
```bash
python detection.py --input test_data/test_scene_2.jpg
```

### Test with your uploaded image:
```bash
# First, save your image as 'my_road_scene.jpg' in my_images folder
python detection.py --input my_images/my_road_scene.jpg
```

## 🎯 What Images Work Best?

### ✅ Good Images:
- Clear road scenes with vehicles
- Images with cars, motorcycles, buses, trucks
- Daylight or well-lit scenes
- Images with pedestrians
- Traffic intersections
- Highway scenes

### ❌ Avoid:
- Very dark or blurry images
- Images without vehicles/people
- Extremely small images (< 200x200 pixels)
- Corrupted or damaged files

## 🔧 Troubleshooting

### "Could not load image" error:
1. Check file path is correct
2. Ensure image format is supported (JPG, PNG, BMP)
3. Make sure file is not corrupted
4. Try renaming file to remove special characters

### No detections found:
1. Image might be too dark
2. Objects might be too small
3. Try a different road scene image
4. Ensure image has vehicles/people

## 📊 Expected Output

When detection works, you'll see:
```
✓ Detected 5 objects
  - car: 0.892
  - motorcycle: 0.756
  - person: 0.634
  - car: 0.821
  - truck: 0.743
```

Plus a window showing your image with colored bounding boxes around detected objects!

## 🎬 For Your Mentor Demo

1. **Prepare 2-3 good road scene images** in `my_images` folder
2. **Name them clearly**: `demo1.jpg`, `demo2.jpg`, `demo3.jpg`
3. **Test each one** before the presentation:
   ```bash
   python detection.py --input my_images/demo1.jpg
   python detection.py --input my_images/demo2.jpg
   python detection.py --input my_images/demo3.jpg
   ```
4. **Choose the best result** for your live demo