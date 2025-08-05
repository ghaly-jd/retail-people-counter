#!/usr/bin/env python3
"""
Download Mall Surveillance Dataset for people counting demo
"""

import requests
import os
import sys
from urllib.parse import urlparse
import zipfile


def download_file(url, filename):
    """Download a file with progress bar"""
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        print(f"\n✓ Downloaded {filename} successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}")
        return False


def extract_zip(zip_path, extract_to="."):
    """Extract zip file"""
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Failed to extract {zip_path}: {e}")
        return False


def download_mall_dataset():
    """Download the Mall Surveillance Dataset"""
    print("🎥 Mall Surveillance Dataset Downloader")
    print("=" * 50)
    
    # Primary Mall dataset URL from CUHK
    mall_urls = [
        "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip",
        "http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip",
    ]
    
    # Alternative sample videos that work well for people counting
    alternative_urls = [
        {
            "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "filename": "sample_mall_video.mp4",
            "description": "Sample video for testing"
        },
        {
            "url": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4", 
            "filename": "sample_people_video.mp4",
            "description": "Sample people video"
        }
    ]
    
    # Try to download Mall dataset
    for i, url in enumerate(mall_urls):
        print(f"Trying Mall dataset URL {i+1}...")
        if download_file(url, "mall_dataset.zip"):
            if extract_zip("mall_dataset.zip"):
                print("🎉 Mall dataset downloaded and extracted successfully!")
                return "mall_dataset"
            else:
                print("Failed to extract mall dataset")
        else:
            print("Failed to download from this URL")
    
    print("\n⚠️  Mall dataset download failed. Trying alternative videos...")
    
    # Try alternative videos
    for alt in alternative_urls:
        print(f"Trying {alt['description']}...")
        if download_file(alt["url"], alt["filename"]):
            print(f"✓ Downloaded {alt['filename']} for demo")
            return alt["filename"]
    
    print("\n❌ All download attempts failed.")
    print("You can manually download a video and place it in this directory.")
    print("Suggested sources:")
    print("1. Mall Dataset: https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html")
    print("2. Any video with people walking/moving")
    return None


def find_video_files():
    """Find video files in current directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    return video_files


if __name__ == "__main__":
    # Check if we already have video files
    existing_videos = find_video_files()
    
    if existing_videos:
        print("🎥 Found existing video files:")
        for i, video in enumerate(existing_videos, 1):
            file_size = os.path.getsize(video) / (1024*1024)  # MB
            print(f"  {i}. {video} ({file_size:.1f} MB)")
        
        response = input("\nUse existing video? (y/n): ").lower().strip()
        if response == 'y':
            selected_video = existing_videos[0]
            print(f"Using {selected_video} for demo")
            sys.exit(0)
    
    # Download dataset
    result = download_mall_dataset()
    
    if result:
        print(f"\n🚀 Ready to create demo with: {result}")
        print("Run the people counter with:")
        if result == "mall_dataset":
            print("   python main.py --video mall_dataset/frames/seq_000001.jpg")
        else:
            print(f"   python main.py --video {result}")
    else:
        print("\n💡 Manual download suggestion:")
        print("1. Download any video with people from YouTube or similar")
        print("2. Save it in this directory")
        print("3. Run: python main.py --video your_video.mp4") 