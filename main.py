import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
from sort import *
from collections import defaultdict
import time


class VehicleCounter:
    def __init__(self, video_path, model_path, mask_path=None):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)

        # Vehicle classes with different confidence thresholds
        self.vehicle_classes = {
            'car': 0.3,
            'truck': 0.25,
            'bus': 0.25,
            'motorbike': 0.35,  # Higher threshold for motorbikes (often harder to detect)
        }

        # COCO class names
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                           "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                           "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]

        # Load mask if provided
        self.mask = cv2.imread(mask_path) if mask_path else None

        # Enhanced tracker with better parameters
        self.tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.2)

        # Counting lines (can have multiple)
        self.counting_lines = [
            {'coords': [400, 297, 673, 297], 'name': 'Line1', 'direction': 'horizontal'}
        ]

        # Tracking data
        self.vehicle_counts = defaultdict(int)  # Count by vehicle type
        self.total_count = 0
        self.tracked_ids = set()
        self.vehicle_paths = defaultdict(list)  # Store vehicle trajectories
        self.last_positions = {}  # For direction detection

        # Performance tracking
        self.fps_counter = 0
        self.start_time = time.time()

    def is_valid_vehicle(self, class_name, confidence):
        """Check if detection is a valid vehicle with appropriate confidence"""
        return class_name in self.vehicle_classes and confidence >= self.vehicle_classes[class_name]


    def check_line_crossing(self, center_x, center_y, vehicle_id, line_info):
        """Enhanced line crossing detection with direction"""
        coords = line_info['coords']
        tolerance = 20  # Increased tolerance

        if line_info['direction'] == 'horizontal':
            line_crossed = (coords[0] < center_x < coords[2] and
                            coords[1] - tolerance < center_y < coords[1] + tolerance)
        else:  # vertical
            line_crossed = (coords[1] < center_y < coords[3] and
                            coords[0] - tolerance < center_x < coords[0] + tolerance)

        return line_crossed

    def draw_statistics(self, img):
        """Draw comprehensive statistics on the image"""
        y_offset = 50

        # Total count
        cvzone.putTextRect(img, f'Total: {self.total_count}', (50, y_offset),
                           scale=1.5, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 0))
        y_offset += 40

        # Count by vehicle type
        for vehicle_type, count in self.vehicle_counts.items():
            if count > 0:
                cvzone.putTextRect(img, f'{vehicle_type.title()}: {count}', (50, y_offset),
                                   scale=1, thickness=2, colorT=(255, 255, 255), colorR=(50, 50, 50))
                y_offset += 30

        # FPS
        elapsed_time = time.time() - self.start_time
        fps = self.fps_counter / elapsed_time if elapsed_time > 0 else 0
        cvzone.putTextRect(img, f'FPS: {fps:.1f}', (img.shape[1] - 150, 50),
                           scale=1, thickness=2, colorT=(0, 255, 0), colorR=(0, 0, 0))

    def process_frame(self):
        """Process a single frame"""
        success, img = self.cap.read()
        if not success:
            return None, False

        self.fps_counter += 1

        # Apply mask if available
        if self.mask is not None:
            img_region = cv2.bitwise_and(img, self.mask)
        else:
            img_region = img

        # Run YOLO detection
        results = self.model(img_region, stream=True, verbose=False)
        detections = np.empty((0, 6))  # Added extra column for class info

        # Process detections
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get confidence and class
                    conf = float(box.conf[0])
                    class_idx = int(box.cls[0])
                    current_class = self.classNames[class_idx]

                    # Filter valid vehicles
                    if self.is_valid_vehicle(current_class, conf):
                        detection_array = np.array([x1, y1, x2, y2, conf, class_idx])
                        detections = np.vstack((detections, detection_array))

        # Update tracker
        if len(detections) > 0:
            results_tracker = self.tracker.update(detections[:, :5])  # Only use first 5 columns for SORT
        else:
            results_tracker = np.empty((0, 5))

        # Draw counting lines
        for line_info in self.counting_lines:
            coords = line_info['coords']
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 3)
            # Add line label
            cv2.putText(img, line_info['name'], (coords[0], coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Process tracked vehicles
        for result in results_tracker:
            x1, y1, x2, y2, track_id = result
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            w, h = x2 - x1, y2 - y1

            # Calculate center point
            center_x, center_y = x1 + w // 2, y1 + h // 2

            # Store trajectory
            self.vehicle_paths[track_id].append((center_x, center_y))
            if len(self.vehicle_paths[track_id]) > 30:  # Keep last 30 points
                self.vehicle_paths[track_id].pop(0)

            # Find corresponding detection to get class info
            vehicle_class = "vehicle"  # Default
            for detection in detections:
                det_x1, det_y1, det_x2, det_y2 = detection[:4].astype(int)
                # Check if tracked object matches detection (with some tolerance)
                if (abs(x1 - det_x1) < 20 and abs(y1 - det_y1) < 20):
                    class_idx = int(detection[5])
                    vehicle_class = self.classNames[class_idx]
                    break

            # Enhanced visualization
            color = (0, 255, 0) if track_id in self.tracked_ids else (255, 0, 255)
            cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=2, colorR=color)

            # Display ID and class
            cvzone.putTextRect(img, f'ID:{track_id} {vehicle_class}',
                               (max(0, x1), max(30, y1)), scale=0.8, thickness=1)

            # Draw center point
            cv2.circle(img, (center_x, center_y), 4, (255, 0, 0), cv2.FILLED)

            # Draw trajectory (last few points)
            if len(self.vehicle_paths[track_id]) > 1:
                points = self.vehicle_paths[track_id][-15:]  # Last 10 points
                for i in range(1, len(points)):
                    # Alpha decreases for older points (0.2 to 1.0)
                    # i=1 (oldest) gets alpha=0.2, i=last (newest) gets alpha=1.0
                    alpha = 0.2 + (i / len(points)) * 0.8

                    overlay = img.copy()
                    cv2.line(overlay, points[i - 1], points[i], (0, 255, 255), 2)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # Check line crossing
            for line_info in self.counting_lines:
                if (self.check_line_crossing(center_x, center_y, track_id, line_info) and
                        track_id not in self.tracked_ids):
                    self.tracked_ids.add(track_id)
                    self.total_count += 1
                    self.vehicle_counts[vehicle_class] += 1

                    # Flash the line green when crossed
                    coords = line_info['coords']
                    cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]),
                             (0, 255, 0), 6)

                    print(f"Vehicle {track_id} ({vehicle_class}) crossed {line_info['name']} - "
                          f"Total: {self.total_count}")

        # Draw statistics
        self.draw_statistics(img)

        return img, True

    def run(self):
        """Main execution loop"""
        print("Starting vehicle counter...")
        print("Press 'q' to quit, 's' to save screenshot")

        while True:
            frame, success = self.process_frame()
            if not success:
                break

            cv2.imshow('Enhanced Vehicle Counter', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"vehicle_count_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"Total vehicles counted: {self.total_count}")
        for vehicle_type, count in self.vehicle_counts.items():
            if count > 0:
                print(f"{vehicle_type.title()}: {count}")


# Usage
if __name__ == "__main__":
    counter = VehicleCounter(
        video_path='../Videos/cars.mp4',
        model_path='../yolo-weights/yolov8l.pt',
        mask_path='mask.png'  # Optional
    )
    counter.run()