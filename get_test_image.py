"""
Download a proper test image with cars, motorcycles, and people
"""

import requests
import os
from pathlib import Path

def download_test_image():
    """Download a test image with multiple vehicles"""
    
    # High-quality traffic image with multiple vehicle types
    test_urls = [
        "https://www.picxy.com/photo/311357",  # Traffic scene
        "https://images.picxy.com/cache/2019/10/31/478d2f1538fa87e380f2a423731ed516.jpg",  # Indian traffic
        "https://cdn.pixabay.com/photo/2016/11/18/15/03/automobile-1835506_960_720.jpg"  # Road scene
    ]
    
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    for i, url in enumerate(test_urls):
        try:
            print(f"Downloading test image {i+1}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            image_path = test_dir / f"test_scene_{i+1}.jpg"
            
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {image_path}")
            
        except Exception as e:
            print(f"❌ Failed to download image {i+1}: {e}")

if __name__ == "__main__":
    download_test_image()
    print("\n🎯 Now run: python detection.py --input test_data/test_scene_1.jpg")