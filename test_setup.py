#!/usr/bin/env python3
"""
Test script to verify that all dependencies are installed correctly
"""

def test_imports():
    """Test if all required packages can be imported"""
    
    print("🧪 Testing Dependencies...")
    print("=" * 50)
    
    tests = [
        ("OpenCV", "import cv2; print(f'OpenCV version: {cv2.__version__}')"),
        ("NumPy", "import numpy as np; print(f'NumPy version: {np.__version__}')"),
        ("Ultralytics (YOLOv8)", "from ultralytics import YOLO; print('YOLOv8 import successful')"),
        ("Requests", "import requests; print(f'Requests version: {requests.__version__}')"),
        ("Collections", "from collections import defaultdict; print('Collections import successful')"),
        ("ArgParse", "import argparse; print('ArgParse import successful')"),
        ("OS", "import os; print('OS import successful')")
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_code in tests:
        try:
            print(f"✓ Testing {test_name}...")
            exec(test_code)
            passed += 1
        except Exception as e:
            print(f"✗ Failed {test_name}: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All dependencies are working correctly!")
        print("🚀 You can now run the people counter with:")
        print("   python main.py --download-sample")
    else:
        print("❌ Some dependencies failed. Please install missing packages:")
        print("   pip install -r requirements.txt")
    
    return failed == 0


def test_yolo_model():
    """Test YOLOv8 model download and basic functionality"""
    try:
        print("\n🤖 Testing YOLOv8 Model...")
        print("=" * 50)
        
        from ultralytics import YOLO
        import numpy as np
        
        # This will download the model if not present
        print("Initializing YOLOv8 model (this may download the model file)...")
        model = YOLO('yolov8n.pt')
        
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Run inference
        print("Running test inference...")
        results = model(dummy_image, verbose=False)
        
        print("✓ YOLOv8 model is working correctly!")
        print(f"✓ Model loaded: {model.model_name if hasattr(model, 'model_name') else 'yolov8n'}")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLOv8 test failed: {e}")
        return False


if __name__ == "__main__":
    print("🛍️ Retail People Counter - Setup Test")
    print("=" * 50)
    
    # Test basic imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test YOLOv8 specifically
        model_ok = test_yolo_model()
        
        if model_ok:
            print("\n🎯 Setup Complete!")
            print("Your people counter is ready to use!")
        else:
            print("\n⚠️  YOLOv8 model test failed.")
            print("The basic imports work, but there might be an issue with the YOLO model.")
    else:
        print("\n❌ Setup incomplete. Please fix dependency issues first.") 