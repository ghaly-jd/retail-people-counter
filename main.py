"""
🛍️ Retail People Counter - Real-time people counting using YOLOv8 and OpenCV
Enhanced with flexible line orientation and optimized tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
import requests
from collections import defaultdict


class PeopleCounter:
    def __init__(self, model_path="yolov8l.pt", confidence_threshold=0.3, line_orientation="horizontal"):
        """
        Initialize the People Counter
        
        Args:
            model_path (str): Path to YOLOv8 model
            confidence_threshold (float): Confidence threshold for detections
            line_orientation (str): "horizontal" for vertical movement, "vertical" for horizontal movement
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.line_orientation = line_orientation
        print(f"🎯 Line orientation: {line_orientation} ({'vertical movement' if line_orientation == 'horizontal' else 'horizontal movement'})")
        
        # Counting variables
        self.people_count = 0
        self.entry_count = 0
        self.exit_count = 0
        
        # Tracking variables
        self.tracked_people = {}
        self.next_id = 1
        
        # Line coordinates (will be set based on video dimensions)
        self.entry_line = None
        
        # Colors for visualization
        self.line_color = (0, 255, 0)  # Green for line
        self.bbox_color = (255, 0, 0)  # Blue for bounding boxes
        
        # Detection history for tracking
        self.detection_history = defaultdict(list)
        
    def set_entry_line(self, video_width, video_height):
        """
        Set entry line based on orientation
        """
        if self.line_orientation == "horizontal":
            # Horizontal line for vertical movement (up/down)
            y = video_height // 3
            self.entry_line = {
                'start': (0, y),
                'end': (video_width, y),
                'coordinate': y,
                'type': 'horizontal'
            }
            print(f"📍 Horizontal entry line set at y={y} (for vertical movement)")
        else:
            # Vertical line for horizontal movement (left/right)
            x = video_width // 3
            self.entry_line = {
                'start': (x, 0),
                'end': (x, video_height),
                'coordinate': x,
                'type': 'vertical'
            }
            print(f"📍 Vertical entry line set at x={x} (for horizontal movement)")
        
    def calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
    
    def detect_people(self, frame):
        """Detect people in the frame using YOLOv8"""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only keep person detections (class_id = 0) above threshold
                    if class_id == 0 and confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append([int(x1), int(y1), int(x2), int(y2)])
        
        return detections
    
    def simple_tracking(self, detections, frame_id):
        """Simple tracking using proximity between frames"""
        current_centers = [self.calculate_center(bbox) for bbox in detections]
        
        # Update existing tracks
        for person_id, history in list(self.tracked_people.items()):
            if len(history) == 0:
                continue
                
            last_center = history[-1]['center']
            min_dist = float('inf')
            best_match_idx = -1
            
            # Find closest detection
            for i, center in enumerate(current_centers):
                dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                if dist < min_dist and dist < 100:  # Distance threshold
                    min_dist = dist
                    best_match_idx = i
            
            # Update track if match found
            if best_match_idx >= 0:
                bbox = detections[best_match_idx]
                center = current_centers[best_match_idx]
                self.tracked_people[person_id].append({
                    'center': center,
                    'bbox': bbox,
                    'frame': frame_id
                })
                # Remove used detection
                detections.pop(best_match_idx)
                current_centers.pop(best_match_idx)
        
        # Create new tracks for remaining detections
        for i, bbox in enumerate(detections):
            if i < len(current_centers):
                center = current_centers[i]
                self.tracked_people[self.next_id] = [{
                    'center': center,
                    'bbox': bbox,
                    'frame': frame_id
                }]
                self.next_id += 1
        
        # Clean old tracks
        self.clean_old_tracks(frame_id)
    
    def clean_old_tracks(self, current_frame, max_age=30):
        """Remove tracks that haven't been updated recently"""
        to_remove = []
        for person_id, history in self.tracked_people.items():
            if len(history) > 0 and current_frame - history[-1]['frame'] > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.tracked_people[person_id]
    
    def check_line_crossing(self, person_id):
        """
        Check if a person has crossed the entry line based on orientation
        """
        if person_id not in self.tracked_people or len(self.tracked_people[person_id]) < 2:
            return None
            
        history = self.tracked_people[person_id]
        prev_center = history[-2]['center']
        curr_center = history[-1]['center']
        
        line_coordinate = self.entry_line['coordinate']
        
        if self.line_orientation == "horizontal":
            # Horizontal line - check y-coordinate crossing (vertical movement)
            if prev_center[1] < line_coordinate and curr_center[1] >= line_coordinate:
                return 'entry'  # Moving downward (top to bottom)
            elif prev_center[1] > line_coordinate and curr_center[1] <= line_coordinate:
                return 'exit'   # Moving upward (bottom to top)
        else:
            # Vertical line - check x-coordinate crossing (horizontal movement)
            if prev_center[0] < line_coordinate and curr_center[0] >= line_coordinate:
                return 'entry'  # Moving rightward (left to right)
            elif prev_center[0] > line_coordinate and curr_center[0] <= line_coordinate:
                return 'exit'   # Moving leftward (right to left)
        
        return None
    
    def update_counts(self):
        """Update entry/exit counts based on line crossings"""
        for person_id in self.tracked_people:
            crossing = self.check_line_crossing(person_id)
            if crossing == 'entry':
                self.entry_count += 1
                self.people_count += 1
            elif crossing == 'exit':
                self.exit_count += 1
                self.people_count -= 1
    
    def draw_visualizations(self, frame, detections):
        """Draw all visualizations on the frame"""
        # Draw entry line
        if self.entry_line:
            cv2.line(frame, self.entry_line['start'], self.entry_line['end'], self.line_color, 3)
            
            # Add line label
            if self.line_orientation == "horizontal":
                label_pos = (10, self.entry_line['coordinate'] - 10)
            else:
                label_pos = (self.entry_line['coordinate'] + 10, 30)
            
            cv2.putText(frame, "ENTRY LINE", label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.line_color, 2)
        
        # Draw bounding boxes
        for bbox in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw counts
        self.draw_counts(frame)
        
        return frame
    
    def draw_counts(self, frame):
        """Draw count information on frame"""
        height, width = frame.shape[:2]
        
        # Background rectangle for counts
        cv2.rectangle(frame, (10, 10), (450, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 120), (255, 255, 255), 2)
        
        # Entry/Exit Flow display (more accurate)
        net_flow = self.entry_count - self.exit_count
        flow_sign = "+" if net_flow >= 0 else ""
        cv2.putText(frame, f"Net Flow: {flow_sign}{net_flow} people", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Entries: {self.entry_count} | Exits: {self.exit_count}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Entry/Exit Flow Tracking", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process video file and count people
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            display (bool): Whether to display video during processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Set entry line
        self.set_entry_line(width, height)
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect people
            detections = self.detect_people(frame)
            
            # Update tracking
            self.simple_tracking(detections, frame_count)
            
            # Update counts
            self.update_counts()
            
            # Draw visualizations
            frame = self.draw_visualizations(frame, detections)
            
            # Write frame if output specified
            if output_path:
                out.write(frame)
            
            # Display frame
            if display:
                cv2.imshow('People Counter', frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Current Count: {self.people_count}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*50)
        print("FINAL STATISTICS - ENTRY/EXIT FLOW")
        print("="*50)
        print(f"Total Entries: {self.entry_count}")
        print(f"Total Exits: {self.exit_count}")
        net_flow = self.entry_count - self.exit_count
        flow_sign = "+" if net_flow >= 0 else ""
        print(f"Net Flow: {flow_sign}{net_flow} people")
        print(f"Frames Processed: {frame_count}")
        print("Note: Net flow shows change during video period only")
        print("="*50)

def download_sample_video():
    """Download a sample video for testing"""
    url = "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
    filename = "sample_video.mp4"
    
    try:
        print("Downloading sample video...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Sample video downloaded: {filename}")
        return filename
    except Exception as e:
        print(f"Failed to download sample video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='People Counter using YOLOv8')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolov8l.pt', 
                       help='YOLOv8 model path (default: yolov8l.pt)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam instead of video file')
    parser.add_argument('--line', type=str, choices=['horizontal', 'vertical'], default='horizontal',
                       help='Line orientation: horizontal (for vertical movement) or vertical (for horizontal movement)')
    parser.add_argument('--download-sample', action='store_true',
                       help='Download sample video for testing')
    
    args = parser.parse_args()
    
    # Download sample video if requested
    if args.download_sample:
        video_path = download_sample_video()
        if not video_path:
            return
    elif args.webcam:
        video_path = 0  # Webcam
    elif args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found")
            return
    else:
        print("Please provide --video, --webcam, or --download-sample")
        return
    
    # Initialize people counter
    print("Initializing People Counter...")
    counter = PeopleCounter(model_path=args.model, confidence_threshold=args.confidence, line_orientation=args.line)
    
    # Process video
    print("Starting people counting...")
    counter.process_video(video_path, args.output)


if __name__ == "__main__":
    main() 