"""
🛍️ Retail People Counter - CLEAN PROFESSIONAL VERSION
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from collections import defaultdict
import time


class CleanPeopleCounter:
    def __init__(self, model_path="yolov8s.pt", confidence_threshold=0.5):
        """
        Initialize the Clean People Counter with professional visualization
        """
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Counting variables
        self.people_count = 0
        self.entry_count = 0
        self.exit_count = 0
        self.max_people_seen = 0
        
        # Tracking variables - optimized for precision
        self.tracked_people = {}
        self.next_id = 0
        self.max_tracking_distance = 80  # Reduced for better precision
        self.track_max_age = 30  # Shorter age for cleaner tracking
        
        # Virtual line parameters
        self.entry_line = None
        self.line_thickness = 2
        self.line_color = (0, 255, 0)  # Green
        
        # Frame tracking
        self.frame_count = 0
        
        # Clean visualization settings
        self.show_tracking_trails = False  # Disabled for clean look
        self.show_confidence = False       # Simplified display
        self.compact_ui = True            # Minimal UI
        
    def set_entry_line(self, video_width, video_height):
        """Set horizontal entry line for mall corridor"""
        y = video_height // 2
        self.entry_line = {
            'start': (0, y),
            'end': (video_width, y),
            'y': y,
            'orientation': 'horizontal'
        }
        
    def calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def detect_people(self, frame):
        """Detect people with improved filtering for precision"""
        results = self.model(frame, verbose=False)
        people_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0 and float(box.conf) > self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        
                        # More strict filtering for precision
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect_ratio = height / width if width > 0 else 0
                        
                        # Filter for human-like proportions and size
                        if (area > 1200 and area < 50000 and  # Reasonable size
                            aspect_ratio > 1.2 and aspect_ratio < 4.0 and  # Human proportions
                            width > 20 and height > 40):  # Minimum dimensions
                            people_detections.append([int(x1), int(y1), int(x2), int(y2), confidence])
        
        return people_detections
    
    def precision_tracking(self, detections, frame_id):
        """Precision tracking with stricter matching"""
        current_centers = [self.calculate_center(det[:4]) for det in detections]
        
        # Match existing tracks with new detections
        matched_tracks = set()
        matched_detections = set()
        
        for person_id, history in list(self.tracked_people.items()):
            if len(history) > 0:
                last_center = history[-1]['center']
                last_bbox = history[-1]['bbox']
                
                best_match = -1
                best_score = float('inf')
                
                for i, (center, detection) in enumerate(zip(current_centers, detections)):
                    if i not in matched_detections:
                        # Distance-based matching
                        distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                        
                        # Size consistency check
                        new_bbox = detection[:4]
                        size_consistency = self.calculate_size_consistency(last_bbox, new_bbox)
                        
                        # Combined score (distance + size consistency)
                        score = distance + (1 - size_consistency) * 50
                        
                        if distance < self.max_tracking_distance and score < best_score:
                            best_score = score
                            best_match = i
                
                if best_match != -1:
                    matched_tracks.add(person_id)
                    matched_detections.add(best_match)
                    
                    # Update track with smoothing
                    self.tracked_people[person_id].append({
                        'center': current_centers[best_match],
                        'bbox': detections[best_match][:4],
                        'confidence': detections[best_match][4],
                        'frame': frame_id
                    })
        
        # Add new tracks for unmatched detections
        for i, (center, detection) in enumerate(zip(current_centers, detections)):
            if i not in matched_detections:
                self.tracked_people[self.next_id] = [{
                    'center': center,
                    'bbox': detection[:4],
                    'confidence': detection[4],
                    'frame': frame_id
                }]
                self.next_id += 1
        
        # Clean old tracks
        self.clean_old_tracks(frame_id)
        
        # Update people count - only count stable tracks
        stable_tracks = 0
        for track in self.tracked_people.values():
            if (len(track) >= 3 and  # Track must exist for at least 3 frames
                frame_id - track[-1]['frame'] < 10):  # Recently updated
                stable_tracks += 1
        
        self.people_count = stable_tracks
        self.max_people_seen = max(self.max_people_seen, self.people_count)
    
    def calculate_size_consistency(self, bbox1, bbox2):
        """Calculate how consistent two bounding boxes are in size"""
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        
        area1, area2 = w1 * h1, w2 * h2
        if area1 == 0 or area2 == 0:
            return 0
        
        ratio = min(area1, area2) / max(area1, area2)
        return ratio
    
    def clean_old_tracks(self, current_frame):
        """Remove old tracks more aggressively for cleaner counting"""
        to_remove = []
        for person_id, history in self.tracked_people.items():
            if (len(history) > 0 and 
                current_frame - history[-1]['frame'] > self.track_max_age):
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.tracked_people[person_id]
    
    def check_line_crossing(self, person_id):
        """Precise line crossing detection with minimum track length"""
        if (person_id not in self.tracked_people or 
            len(self.tracked_people[person_id]) < 5):  # Require longer track for precision
            return None
            
        history = self.tracked_people[person_id]
        
        # Check last few positions for consistent crossing
        y_positions = [h['center'][1] for h in history[-5:]]
        line_y = self.entry_line['y']
        
        # Look for clear crossing pattern
        above_line = [y_pos < line_y for y_pos in y_positions]
        
        if len(above_line) >= 4:
            # Check for clear transition
            if above_line[0] and above_line[1] and not above_line[-2] and not above_line[-1]:
                return 'entry'  # Clear entry pattern
            elif not above_line[0] and not above_line[1] and above_line[-2] and above_line[-1]:
                return 'exit'   # Clear exit pattern
        
        return None
    
    def update_counts(self):
        """Update counts with anti-duplicate measures"""
        for person_id in list(self.tracked_people.keys()):
            if len(self.tracked_people[person_id]) >= 5:  # Only check established tracks
                crossing = self.check_line_crossing(person_id)
                
                # Mark this track as counted to prevent re-counting
                if crossing and not hasattr(self.tracked_people[person_id][-1], 'counted'):
                    if crossing == 'entry':
                        self.entry_count += 1
                    elif crossing == 'exit':
                        self.exit_count += 1
                    
                    # Mark as counted
                    self.tracked_people[person_id][-1]['counted'] = True
    
    def draw_clean_visualizations(self, frame, detections):
        """Clean, professional visualization"""
        # Draw entry line (thinner, less intrusive)
        if self.entry_line:
            cv2.line(frame, self.entry_line['start'], self.entry_line['end'], 
                    self.line_color, self.line_thickness)
            
            # Small line label
            cv2.putText(frame, "Counting Line", 
                       (10, self.entry_line['y'] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.line_color, 1)
        
        # Draw clean bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 0.0
            
            # Simple, clean bounding boxes
            if confidence > 0.6:
                color = (0, 255, 0)  # Green for high confidence
            else:
                color = (0, 255, 255)  # Yellow for medium confidence
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            # Optional: small confidence label (only for high confidence)
            if self.show_confidence and confidence > 0.7:
                cv2.putText(frame, f"{confidence:.2f}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # NO tracking trails for clean look
        
        # Compact information panel
        self.draw_compact_info(frame)
        
        return frame
    
    def draw_compact_info(self, frame):
        """Compact, professional information display"""
        height, width = frame.shape[:2]
        
        # Small, compact info panel
        panel_width = 200
        panel_height = 80
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), (255, 255, 255), 1)
        
        # Compact text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Current count (most important)
        cv2.putText(frame, f"People: {self.people_count}", 
                   (20, 30), font, 0.6, (0, 255, 0), thickness + 1)
        
        # Entries and exits
        cv2.putText(frame, f"Entries: {self.entry_count}", 
                   (20, 50), font, font_scale, (0, 255, 255), thickness)
        
        cv2.putText(frame, f"Exits: {self.exit_count}", 
                   (20, 70), font, font_scale, (0, 165, 255), thickness)
        
        # Frame number (small, bottom right)
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (width - 100, height - 10), font, 0.3, (255, 255, 255), 1)
    
    def process_video(self, video_path, output_path=None, display=True, max_frames=None):
        """Process video with clean, professional output"""
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
        
        print(f"Creating CLEAN professional demo:")
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"Model: YOLOv8s with confidence {self.confidence_threshold}")
        
        # Set entry line
        self.set_entry_line(width, height)
        
        # Setup video writer
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
            self.precision_tracking(detections, self.frame_count)
            
            # Update counts
            self.update_counts()
            
            # Draw clean visualizations
            frame = self.draw_clean_visualizations(frame, detections)
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Display
            if display:
                cv2.imshow('Professional People Counter', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
            
            self.frame_count += 1
            
            # Progress (less frequent for cleaner output)
            if self.frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = self.frame_count / elapsed
                progress = (self.frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | FPS: {fps_actual:.1f} | People: {self.people_count}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Final professional statistics
        print("\n" + "="*50)
        print("🎯 PROFESSIONAL DEMO STATISTICS")
        print("="*50)
        print(f"📊 Final People Count: {self.people_count}")
        print(f"📈 Maximum Occupancy: {self.max_people_seen}")
        print(f"🚶 Total Entries: {self.entry_count}")
        print(f"🚪 Total Exits: {self.exit_count}")
        print(f"🎬 Frames Processed: {self.frame_count}")
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Processing Time: {elapsed_time:.2f} seconds")
        print(f"🚀 Average FPS: {self.frame_count / elapsed_time:.2f}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Clean Professional People Counter')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolov8s.pt', 
                       help='YOLOv8 model path')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam')
    parser.add_argument('--max-frames', type=int, 
                       help='Maximum frames to process')
    
    args = parser.parse_args()
    
    if args.webcam:
        video_path = 0
    elif args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found")
            return
    else:
        print("Please provide --video or --webcam")
        return
    
    # Initialize clean counter
    print("🎬 Initializing Professional People Counter...")
    counter = CleanPeopleCounter(model_path=args.model, confidence_threshold=args.confidence)
    
    # Process video
    print("🚀 Creating professional demo...")
    counter.process_video(video_path, args.output, max_frames=args.max_frames)


if __name__ == "__main__":
    main() 