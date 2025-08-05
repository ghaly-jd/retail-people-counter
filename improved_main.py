"""
🛍️ Retail People Counter - IMPROVED VERSION with better tracking and debugging
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from collections import defaultdict
import time


class ImprovedPeopleCounter:
    def __init__(self, model_path="yolov8s.pt", confidence_threshold=0.4):
        """
        Initialize the Improved People Counter
        
        Args:
            model_path (str): Path to YOLOv8 model (using yolov8s for better accuracy)
            confidence_threshold (float): Lower threshold for better detection
        """
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Counting variables - FIXED: Don't reset current count on crossings
        self.people_count = 0
        self.entry_count = 0
        self.exit_count = 0
        self.max_people_seen = 0
        
        # Tracking variables with better parameters
        self.tracked_people = {}
        self.next_id = 0
        self.max_tracking_distance = 120  # Increased for better tracking
        self.track_max_age = 60  # Keep tracks longer
        
        # Virtual line parameters - IMPROVED positioning
        self.entry_line = None
        self.line_thickness = 4
        self.line_color = (0, 255, 0)  # Green
        
        # Detection history for better tracking
        self.detection_history = defaultdict(list)
        self.frame_count = 0
        
        # Debug info
        self.debug_mode = True
        self.detection_stats = {"detections_per_frame": []}
        
    def set_entry_line(self, video_width, video_height):
        """
        IMPROVED: Set the virtual entry line - better positioned for mall footage
        """
        # For mall dataset: horizontal line works better than vertical
        # People walk up/down in the corridor
        y = video_height // 2  # Horizontal line in middle
        self.entry_line = {
            'start': (0, y),
            'end': (video_width, y),
            'y': y,
            'orientation': 'horizontal'
        }
        
        print(f"Entry line set: horizontal at y={y}")
        
    def calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def detect_people(self, frame):
        """
        IMPROVED: Detect people with better filtering
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
                        confidence = float(box.conf)
                        
                        # Filter out very small detections (likely false positives)
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        if area > 800:  # Minimum area threshold
                            people_detections.append([int(x1), int(y1), int(x2), int(y2), confidence])
        
        # Store detection stats
        self.detection_stats["detections_per_frame"].append(len(people_detections))
        
        return people_detections
    
    def improved_tracking(self, detections, frame_id):
        """
        IMPROVED: Better tracking algorithm with Hungarian algorithm-like matching
        """
        current_centers = [self.calculate_center(det[:4]) for det in detections]
        
        # Calculate distance matrix between existing tracks and new detections
        matched_tracks = set()
        matched_detections = set()
        
        for person_id, history in list(self.tracked_people.items()):
            if len(history) > 0:
                last_center = history[-1]['center']
                
                best_match = -1
                best_distance = float('inf')
                
                for i, center in enumerate(current_centers):
                    if i not in matched_detections:
                        distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                        
                        # Consider velocity prediction for better matching
                        if len(history) >= 2:
                            prev_center = history[-2]['center']
                            velocity = (last_center[0] - prev_center[0], last_center[1] - prev_center[1])
                            predicted_center = (last_center[0] + velocity[0], last_center[1] + velocity[1])
                            predicted_distance = np.sqrt((center[0] - predicted_center[0])**2 + (center[1] - predicted_center[1])**2)
                            distance = min(distance, predicted_distance)
                        
                        if distance < self.max_tracking_distance and distance < best_distance:
                            best_distance = distance
                            best_match = i
                
                if best_match != -1:
                    matched_tracks.add(person_id)
                    matched_detections.add(best_match)
                    
                    # Update track
                    self.tracked_people[person_id].append({
                        'center': current_centers[best_match],
                        'bbox': detections[best_match][:4],
                        'confidence': detections[best_match][4] if len(detections[best_match]) > 4 else 0.5,
                        'frame': frame_id
                    })
        
        # Add unmatched detections as new tracks
        for i, (center, detection) in enumerate(zip(current_centers, detections)):
            if i not in matched_detections:
                self.tracked_people[self.next_id] = [{
                    'center': center,
                    'bbox': detection[:4],
                    'confidence': detection[4] if len(detection) > 4 else 0.5,
                    'frame': frame_id
                }]
                self.next_id += 1
        
        # Clean old tracks
        self.clean_old_tracks(frame_id)
        
        # Update current people count based on active tracks
        active_tracks = len([track for track in self.tracked_people.values() 
                           if len(track) > 0 and frame_id - track[-1]['frame'] < 30])
        
        # IMPROVED: More stable people counting
        self.people_count = active_tracks
        self.max_people_seen = max(self.max_people_seen, self.people_count)
    
    def clean_old_tracks(self, current_frame, max_age=None):
        """Remove tracks that haven't been updated recently"""
        if max_age is None:
            max_age = self.track_max_age
            
        to_remove = []
        for person_id, history in self.tracked_people.items():
            if len(history) > 0 and current_frame - history[-1]['frame'] > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.tracked_people[person_id]
    
    def check_line_crossing(self, person_id):
        """
        IMPROVED: Check if a person has crossed the entry line (horizontal)
        """
        if person_id not in self.tracked_people or len(self.tracked_people[person_id]) < 2:
            return None
            
        history = self.tracked_people[person_id]
        prev_center = history[-2]['center']
        curr_center = history[-1]['center']
        
        line_y = self.entry_line['y']
        
        # Check if crossed the horizontal line
        if prev_center[1] < line_y and curr_center[1] >= line_y:
            return 'entry'  # Moving down (entering)
        elif prev_center[1] > line_y and curr_center[1] <= line_y:
            return 'exit'   # Moving up (exiting)
        
        return None
    
    def update_counts(self):
        """IMPROVED: Update entry/exit counts based on line crossings"""
        for person_id in list(self.tracked_people.keys()):
            crossing = self.check_line_crossing(person_id)
            if crossing == 'entry':
                self.entry_count += 1
                if self.debug_mode:
                    print(f"Frame {self.frame_count}: ENTRY detected (ID: {person_id})")
            elif crossing == 'exit':
                self.exit_count += 1
                if self.debug_mode:
                    print(f"Frame {self.frame_count}: EXIT detected (ID: {person_id})")
    
    def draw_improved_visualizations(self, frame, detections):
        """
        IMPROVED: Draw enhanced visualizations with debug info
        """
        # Draw entry line with better visibility
        if self.entry_line:
            cv2.line(frame, self.entry_line['start'], self.entry_line['end'], 
                    self.line_color, self.line_thickness)
            cv2.putText(frame, "COUNTING LINE", 
                       (10, self.entry_line['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.line_color, 2)
        
        # Draw bounding boxes with confidence and tracking IDs
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 0.0
            
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracking trails for active tracks
        for person_id, history in self.tracked_people.items():
            if len(history) > 1:
                points = [h['center'] for h in history[-10:]]  # Last 10 positions
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (255, 0, 255), 2)
        
        # Draw enhanced counts and statistics
        self.draw_enhanced_counts(frame, detections)
        
        return frame
    
    def draw_enhanced_counts(self, frame, detections):
        """IMPROVED: Draw enhanced count information with statistics"""
        height, width = frame.shape[:2]
        
        # Larger background rectangle for more info
        cv2.rectangle(frame, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, 200), (255, 255, 255), 3)
        
        # Count text with better formatting
        y_offset = 35
        line_height = 25
        
        # Current stats
        cv2.putText(frame, f"Current People: {self.people_count}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Max Seen: {self.max_people_seen}", 
                   (20, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Total Entries: {self.entry_count}", 
                   (20, y_offset + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Total Exits: {self.exit_count}", 
                   (20, y_offset + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        cv2.putText(frame, f"Active Tracks: {len(self.tracked_people)}", 
                   (20, y_offset + 4*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Detections: {len(detections)}", 
                   (20, y_offset + 5*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_video(self, video_path, output_path=None, display=True, max_frames=None):
        """
        IMPROVED: Process video with better progress tracking and debugging
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
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"Using model: {self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8'}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # Set entry line
        self.set_entry_line(width, height)
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.frame_count = 0
        start_time = time.time()
        
        while self.frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect people
            detections = self.detect_people(frame)
            
            # Update tracking
            self.improved_tracking(detections, self.frame_count)
            
            # Update counts
            self.update_counts()
            
            # Draw visualizations
            frame = self.draw_improved_visualizations(frame, detections)
            
            # Write frame if output specified
            if output_path:
                out.write(frame)
            
            # Display frame
            if display:
                cv2.imshow('Improved People Counter', frame)
                
                # Exit on 'q' key, pause on 'p'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)  # Pause until any key
            
            self.frame_count += 1
            
            # Print progress and statistics
            if self.frame_count % 50 == 0:
                elapsed_time = time.time() - start_time
                fps_actual = self.frame_count / elapsed_time
                progress = (self.frame_count / total_frames) * 100
                avg_detections = np.mean(self.detection_stats["detections_per_frame"][-50:]) if self.detection_stats["detections_per_frame"] else 0
                
                print(f"Progress: {progress:.1f}% | FPS: {fps_actual:.1f} | People: {self.people_count} | Avg Det: {avg_detections:.1f}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final enhanced statistics
        print("\n" + "="*60)
        print("ENHANCED FINAL STATISTICS")
        print("="*60)
        print(f"Total Entries: {self.entry_count}")
        print(f"Total Exits: {self.exit_count}")
        print(f"Final People Count: {self.people_count}")
        print(f"Max People Seen: {self.max_people_seen}")
        print(f"Total Tracks Created: {self.next_id}")
        print(f"Frames Processed: {self.frame_count}")
        
        if self.detection_stats["detections_per_frame"]:
            avg_det = np.mean(self.detection_stats["detections_per_frame"])
            print(f"Average Detections per Frame: {avg_det:.2f}")
        
        elapsed_time = time.time() - start_time
        print(f"Processing Time: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {self.frame_count / elapsed_time:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Improved People Counter using YOLOv8')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolov8s.pt', 
                       help='YOLOv8 model path (default: yolov8s.pt for better accuracy)')
    parser.add_argument('--confidence', type=float, default=0.4,
                       help='Confidence threshold (default: 0.4)')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam instead of video file')
    parser.add_argument('--max-frames', type=int, 
                       help='Maximum frames to process (for testing)')
    
    args = parser.parse_args()
    
    if args.webcam:
        video_path = 0  # Webcam
    elif args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found")
            return
    else:
        print("Please provide --video or --webcam")
        return
    
    # Initialize improved people counter
    print("Initializing Improved People Counter...")
    counter = ImprovedPeopleCounter(model_path=args.model, confidence_threshold=args.confidence)
    
    # Process video
    print("Starting improved people counting...")
    counter.process_video(video_path, args.output, max_frames=args.max_frames)


if __name__ == "__main__":
    main() 