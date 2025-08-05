#!/usr/bin/env python3
"""
Create demo video from Mall Surveillance Dataset using our people counter
"""

import cv2
import os
import glob
import sys
from pathlib import Path
import subprocess


def images_to_video(image_folder, output_video, fps=10):
    """Convert image sequence to video"""
    print(f"Converting images from {image_folder} to video...")
    
    # Get list of images
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    
    if not image_files:
        print("No images found!")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print("Could not read first image!")
        return False
    
    height, width, layers = first_image.shape
    print(f"Video dimensions: {width}x{height}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Add each image to video
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        if img is not None:
            out.write(img)
        
        if i % 50 == 0:
            progress = (i / len(image_files)) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print(f"✓ Video saved as {output_video}")
    return True


def find_mall_images():
    """Find the mall dataset image files"""
    possible_paths = [
        "mall_dataset/frames",
        "mall_dataset",
        "frames",
        "."
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            jpg_files = glob.glob(os.path.join(path, "*.jpg"))
            if jpg_files:
                print(f"Found {len(jpg_files)} images in {path}")
                return path
    
    return None


def create_mall_demo():
    """Create demo video from mall dataset"""
    print("🛍️ Creating Mall Dataset Demo")
    print("=" * 50)
    
    # Find mall images
    image_folder = find_mall_images()
    if not image_folder:
        print("❌ No mall dataset images found!")
        print("Make sure the mall dataset is downloaded and extracted.")
        return False
    
    # Create video from images
    mall_video = "mall_dataset_video.mp4"
    if not images_to_video(image_folder, mall_video, fps=8):
        return False
    
    # Process video with people counter
    print("\n🤖 Processing with People Counter...")
    print("=" * 50)
    
    demo_output = "mall_demo_with_counting.mp4"
    
    # Run our people counter on the mall video
    cmd = [
        sys.executable, "main.py",
        "--video", mall_video,
        "--output", demo_output,
        "--confidence", "0.6"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ People counting completed successfully!")
            print(f"✓ Demo video saved as: {demo_output}")
            
            # Show final statistics from output
            if result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "FINAL STATISTICS" in line or "Total Entries" in line or "Total Exits" in line or "Current People Count" in line:
                        print(line)
            
            return demo_output
        else:
            print(f"❌ People counter failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running people counter: {e}")
        return False


def analyze_demo_results():
    """Analyze and display demo results"""
    demo_file = "mall_demo_with_counting.mp4"
    
    if os.path.exists(demo_file):
        file_size = os.path.getsize(demo_file) / (1024*1024)  # MB
        
        print("\n🎯 Demo Video Created Successfully!")
        print("=" * 50)
        print(f"📁 File: {demo_file}")
        print(f"📊 Size: {file_size:.1f} MB")
        
        # Get video info
        cap = cv2.VideoCapture(demo_file)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"🎬 Resolution: {width}x{height}")
            print(f"⏱️  Duration: {duration:.1f} seconds")
            print(f"🎞️  Frame Rate: {fps:.1f} FPS")
            print(f"📷 Total Frames: {frame_count}")
            
            cap.release()
        
        print("\n🚀 Your demo is ready!")
        print("You can now:")
        print("1. Upload to GitHub")
        print("2. Share on LinkedIn")
        print("3. Include in your portfolio")
        print("4. Submit to job applications")
        
        return True
    else:
        print("❌ Demo video not found")
        return False


if __name__ == "__main__":
    print("🎥 Mall Dataset Demo Creator")
    print("Building professional people counting demo...")
    print()
    
    # Create the demo
    demo_result = create_mall_demo()
    
    if demo_result:
        # Analyze results
        analyze_demo_results()
        
        print("\n🎉 SUCCESS! Your retail people counter demo is complete!")
    else:
        print("\n❌ Demo creation failed. Check the logs above for details.") 