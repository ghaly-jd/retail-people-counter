"""
🛍️ Retail People Counter - Real-time people counting using YOLOv8 and OpenCV
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from collections import defaultdict


class PeopleCounter:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        """
        Initialize the People Counter
        
        Args:
            model_path (str): Path to YOLOv8 model
            confidence_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Counting variables
        self.people_count = 0
        self.entry_count = 0
        self.exit_count = 0
        
        # Tracking variables
        self.tracked_people = {}
        self.next_id = 0
        
        # Virtual line parameters (will be set based on video dimensions)
        self.entry_line = None
        self.line_thickness = 3
        self.line_color = (0, 255, 0)  # Green
        
        # Detection history for tracking
        self.detection_history = defaultdict(list)
        
    def set_entry_line(self, video_width, video_height):
        """
        Set the virtual entry line based on video dimensions
        Default: vertical line at 1/3 from left
        """
        x = video_width // 3
        self.entry_line = {
            'start': (x, 0),
            'end': (x, video_height),
            'x': x
        }
        
    def calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def detect_people(self, frame):
        """
        Detect people in the frame using YOLOv8
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detected person bounding boxes
        """
        results = self.model(frame, verbose=False)
        people_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Class 0 is 'person' in COCO dataset
                    if int(box.cls) == 0 and float(box.conf) > self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        people_detections.append([int(x1), int(y1), int(x2), int(y2)])
        
        return people_detections
    
    def simple_tracking(self, detections, frame_id):
        """
        Simple tracking based on proximity between frames
        """
        current_centers = [self.calculate_center(det) for det in detections]
        
        # Update tracking
        matched = set()
        for person_id, history in self.tracked_people.items():
            if len(history) > 0:
                last_center = history[-1]['center']
                # Find closest detection
                min_dist = float('inf')
                closest_idx = -1
                
                for i, center in enumerate(current_centers):
                    if i not in matched:
                        dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                        if dist < min_dist and dist < 100:  # Max distance threshold
                            min_dist = dist
                            closest_idx = i
                
                if closest_idx != -1:
                    matched.add(closest_idx)
                    self.tracked_people[person_id].append({
                        'center': current_centers[closest_idx],
                        'bbox': detections[closest_idx],
                        'frame': frame_id
                    })
        
        # Add new detections as new people
        for i, (center, bbox) in enumerate(zip(current_centers, detections)):
            if i not in matched:
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
        Check if a person has crossed the entry line
        """
        if person_id not in self.tracked_people or len(self.tracked_people[person_id]) < 2:
            return None
            
        history = self.tracked_people[person_id]
        prev_center = history[-2]['center']
        curr_center = history[-1]['center']
        
        line_x = self.entry_line['x']
        
        # Check if crossed the line
        if prev_center[0] < line_x and curr_center[0] >= line_x:
            return 'entry'
        elif prev_center[0] > line_x and curr_center[0] <= line_x:
            return 'exit'
        
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
                
        # Ensure count doesn't go negative
        self.people_count = max(0, self.people_count)
    
    def draw_visualizations(self, frame, detections):
        """
        Draw bounding boxes, counts, and entry line on frame
        """
        # Draw entry line
        if self.entry_line:
            cv2.line(frame, self.entry_line['start'], self.entry_line['end'], 
                    self.line_color, self.line_thickness)
            cv2.putText(frame, "ENTRY LINE", 
                       (self.entry_line['x'] + 10, 30), 
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
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Count text
        cv2.putText(frame, f"Current People: {self.people_count}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Entries: {self.entry_count}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Total Exits: {self.exit_count}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
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
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total Entries: {self.entry_count}")
        print(f"Total Exits: {self.exit_count}")
        print(f"Current People Count: {self.people_count}")
        print(f"Frames Processed: {frame_count}")


def download_sample_video():
    """Download a sample video for testing"""
    import requests
    
    # Sample video URL (you can replace this with mall dataset video)
    sample_url = "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
    
    print("Downloading sample video...")
    try:
        response = requests.get(sample_url, stream=True)
        if response.status_code == 200:
            with open("sample_video.mp4", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Sample video downloaded: sample_video.mp4")
            return "sample_video.mp4"
        else:
            print("Failed to download sample video")
            return None
    except Exception as e:
        print(f"Error downloading sample video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='People Counter using YOLOv8')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLOv8 model path (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam instead of video file')
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
    counter = PeopleCounter(model_path=args.model, confidence_threshold=args.confidence)
    
    # Process video
    print("Starting people counting...")
    counter.process_video(video_path, args.output)


if __name__ == "__main__":
    main() 