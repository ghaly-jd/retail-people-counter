#!/usr/bin/env python3
"""
Download PETS2009 Dataset for People Counting
Includes multiple download sources and automatic extraction
"""

import os
from pathlib import Path
import sys

class PETS2009Downloader:
    def __init__(self):
        self.base_dir = Path("datasets/PETS2009")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def setup_pets_demo(self):
        """Set up PETS2009 demo configuration"""
        print("🔧 Setting up PETS2009 demo configuration...")
        
        # Create videos directory
        video_dir = self.base_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        
        # Create README with instructions
        readme_content = """# PETS2009 Dataset Setup

## Official Sources:
1. PETS2009 Website: http://www.cvg.reading.ac.uk/PETS2009/
2. Academic Papers: Search "PETS2009 benchmark dataset"

## Manual Download Instructions:
1. Visit the PETS2009 official website  
2. Download desired scenarios (S2L1, S2L2, S2L3 recommended)
3. Place .avi/.mp4 files in: datasets/PETS2009/videos/
4. Run: python datasets/PETS2009/run_pets_demo.py path/to/video.avi

## Recommended Scenarios for People Counting:
- S2L1: Low density walking (easiest)
- S2L2: Medium density walking 
- S2L3: High density walking (challenging)
- S1L1: Single person (baseline)
"""
        
        with open(self.base_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        # Create demo runner script
        demo_script = self.base_dir / "run_pets_demo.py"
        demo_content = '''#!/usr/bin/env python3
"""
PETS2009 Demo Runner
Optimized settings for PETS surveillance footage
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from clean_main import CleanPeopleCounter

def run_pets_demo(video_path, scenario_name="PETS2009"):
    """Run people counter optimized for PETS2009 dataset"""
    
    print(f"🎬 Running PETS2009 Demo: {scenario_name}")
    print(f"📹 Video: {video_path}")
    
    # PETS-optimized settings
    counter = CleanPeopleCounter(
        model_path='yolov8s.pt',  # Better accuracy for surveillance
        confidence_threshold=0.3,  # PETS videos are high quality
        compact_ui=True,
        show_tracking_trails=False,
        debug_mode=False
    )
    
    # Output path
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"pets2009_{scenario_name}_demo.mp4"
    
    # Process video
    counter.process_video(
        video_path=video_path,
        output_path=str(output_path),
        max_frames=3000  # ~2 minutes at 25fps
    )
    
    print(f"✅ PETS2009 demo saved: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pets_demo.py <video_path> [scenario_name]")
        print("Example: python run_pets_demo.py videos/S2L1.avi S2L1_View1")
        sys.exit(1)
    
    video_path = sys.argv[1]
    scenario_name = sys.argv[2] if len(sys.argv) > 2 else "PETS2009"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    run_pets_demo(video_path, scenario_name)
'''
        
        with open(demo_script, 'w') as f:
            f.write(demo_content)
        
        print(f"✅ PETS2009 demo script created: {demo_script}")
        return demo_script

def main():
    print("🐾 PETS2009 Dataset Setup for People Counting")
    print("=" * 50)
    
    downloader = PETS2009Downloader()
    
    # Setup demo runner
    demo_script = downloader.setup_pets_demo()
    
    print("\n🎯 Next Steps:")
    print("1. Download PETS2009 videos from official website")
    print("2. Place videos in datasets/PETS2009/videos/")
    print("3. Run: python datasets/PETS2009/run_pets_demo.py path/to/video.avi")
    
    print("\n💡 Status:")
    print("✅ PETS2009 directory structure created")
    print("✅ Demo runner script ready")
    print("📋 Download instructions available")

if __name__ == "__main__":
    main()
