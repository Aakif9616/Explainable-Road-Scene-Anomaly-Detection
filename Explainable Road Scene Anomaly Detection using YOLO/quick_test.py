"""
Quick test script - automatically finds and tests images in my_images folder
"""

import os
import glob
from pathlib import Path

def find_and_test_images():
    """Find all images in my_images folder and show commands to test them"""
    
    my_images_dir = Path("my_images")
    
    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    print("🔍 Looking for images in my_images folder...")
    print("=" * 50)
    
    found_images = []
    for ext in image_extensions:
        found_images.extend(glob.glob(str(my_images_dir / ext)))
    
    if not found_images:
        print("❌ No images found in my_images folder!")
        print("\n📋 To add images:")
        print("1. Save your road scene image to the 'my_images' folder")
        print("2. Supported formats: JPG, PNG, BMP")
        print("3. Then run this script again")
        print("\n💡 Or try the sample image:")
        print("   python detection.py --input test_data/test_scene_2.jpg")
        return
    
    print(f"✅ Found {len(found_images)} image(s):")
    print()
    
    for i, image_path in enumerate(found_images, 1):
        image_name = os.path.basename(image_path)
        print(f"{i}. {image_name}")
        print(f"   Command: python detection.py --input {image_path}")
        print()
    
    print("🚀 Copy and paste any command above to test detection!")
    print("=" * 50)

if __name__ == "__main__":
    print("🎯 Quick Image Test Helper")
    print("=" * 50)
    find_and_test_images()