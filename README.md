# 🛍️ Retail People Counter

A real-time people counting system for retail stores, gyms, and events using YOLOv8 and OpenCV. This project detects people entering and leaving from video footage and displays real-time counts with clear visualizations.

![People Counter Demo](demo_preview.png)

## 🎯 Features

- **Real-time People Detection**: Uses YOLOv8 for accurate person detection
- **Entry/Exit Counting**: Virtual line-based counting system
- **Visual Feedback**: Clear overlay showing current count, entries, and exits
- **Multiple Input Sources**: Support for video files, webcam, and sample footage
- **Tracking System**: Simple tracking to minimize double-counting
- **Export Capability**: Save processed video with counting overlays

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/ghaly-jd/retail-people-counter.git
cd retail-people-counter
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLOv8 model** (automatic on first run)
```bash
# The script will automatically download yolov8n.pt on first run
# No manual action needed
```

## 🚀 Usage

### Basic Usage

**Process a video file:**
```bash
python main.py --video path/to/your/video.mp4
```

**Use webcam:**
```bash
python main.py --webcam
```

**Download and test with sample video:**
```bash
python main.py --download-sample
```

### Advanced Options

**Save output video with counting overlay:**
```bash
python main.py --video input.mp4 --output counted_output.mp4
```

**Adjust confidence threshold:**
```bash
python main.py --video input.mp4 --confidence 0.7
```

**Use different YOLOv8 model:**
```bash
python main.py --video input.mp4 --model yolov8s.pt
```

### Complete Command Reference

```bash
python main.py [OPTIONS]

Options:
  --video PATH          Path to input video file
  --output PATH         Path to save output video with overlays
  --model PATH          YOLOv8 model path (default: yolov8n.pt)
  --confidence FLOAT    Detection confidence threshold (default: 0.5)
  --webcam             Use webcam as input source
  --download-sample    Download sample video for testing
  --help               Show help message
```

## 📊 How It Works

### 1. People Detection
- Uses YOLOv8 (You Only Look Once) neural network
- Detects people with bounding boxes and confidence scores
- Filters detections based on confidence threshold

### 2. Tracking System
- Simple proximity-based tracking between frames
- Assigns unique IDs to detected people
- Handles people entering/leaving the frame

### 3. Counting Logic
- Virtual entry line positioned at 1/3 of frame width
- Tracks when people cross the line in either direction
- Entry: left-to-right crossing (count +1)
- Exit: right-to-left crossing (count -1)

### 4. Visualization
- Real-time count display in top-left corner
- Green entry line visualization
- Bounding boxes around detected people
- Statistics: current count, total entries, total exits

## 🎥 Mall Surveillance Dataset

For testing with realistic retail footage, download the Mall Surveillance Dataset:

1. **Download**: [Mall Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)
2. **Extract** the video file
3. **Run**: `python main.py --video path/to/mall_dataset.avi`

The dataset contains surveillance footage of people in a mall corridor, perfect for testing retail counting scenarios.

## 📈 Performance Tips

### Accuracy Improvements
- **Adjust confidence threshold**: Higher values (0.7-0.8) reduce false positives
- **Optimal lighting**: Works best with good lighting conditions
- **Camera angle**: Overhead or angled views work better than straight-on
- **Line positioning**: Adjust entry line position for your specific camera setup

### Speed Optimization
- **Use smaller models**: `yolov8n.pt` (nano) for speed, `yolov8s.pt` (small) for balance
- **Reduce resolution**: Resize video frames if processing is slow
- **Skip frames**: Process every 2nd or 3rd frame for faster processing

## 🔧 Customization

### Modify Entry Line Position
Edit the `set_entry_line` method in `main.py`:
```python
def set_entry_line(self, video_width, video_height):
    x = video_width // 2  # Center line instead of 1/3
    self.entry_line = {
        'start': (x, 0),
        'end': (x, video_height),
        'x': x
    }
```

### Adjust Tracking Parameters
Modify tracking sensitivity in the `simple_tracking` method:
```python
if dist < min_dist and dist < 150:  # Increase from 100 for larger movement tolerance
```

### Change Visualization Colors
Update colors in the class initialization:
```python
self.line_color = (0, 0, 255)  # Red instead of green
```

## 📝 Example Output

```
Video Info: 1280x720, 30 FPS, 1500 frames
Progress: 20.0% - Current Count: 3
Progress: 40.0% - Current Count: 5
Progress: 60.0% - Current Count: 2
Progress: 80.0% - Current Count: 4
Progress: 100.0% - Current Count: 3

==================================================
FINAL STATISTICS
==================================================
Total Entries: 28
Total Exits: 25
Current People Count: 3
Frames Processed: 1500
```

## 🧪 Testing

Run the setup test to verify installation:
```bash
python test_setup.py
```

This will check all dependencies and download the YOLOv8 model if needed.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv8 implementation
- **OpenCV** community for computer vision tools
- **CUHK** for the Mall Surveillance Dataset
- **COCO** dataset for training the person detection model

## 🔗 Links

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Mall Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)

---

**Built with ❤️ for retail analytics and computer vision learning** 