"""
Sample image downloader for testing the road scene detection system
"""

import requests
import os
from pathlib import Path

def download_sample_image():
    """Download a sample Indian traffic image for testing"""
    
    # Sample Indian traffic image URL (free to use from Pixabay)
    sample_url = "https://cdn.pixabay.com/photo/2017/08/06/12/06/people-2591874_960_720.jpg"
    
    # Create test_data directory if it doesn't exist
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Download the image
    image_path = test_dir / "indian_road_scene.jpg"
    
    try:
        print("Downloading sample traffic image...")
        response = requests.get(sample_url, stream=True)
        response.raise_for_status()
        
        with open(image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Indian road scene image downloaded: {image_path}")
        return str(image_path)
        
    except Exception as e:
        print(f"❌ Failed to download sample image: {e}")
        return None

if __name__ == "__main__":
    download_sample_image()