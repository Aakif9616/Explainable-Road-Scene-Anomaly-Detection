"""
Enhanced Explainable Road Scene Anomaly Detection System
Academic Demonstration - 40% Implementation

This system demonstrates:
1. Data Collection Module - Load traffic images/videos
2. Preprocessing Module - Resize and normalize frames
3. Detection Module - YOLO-based vehicle/pedestrian detection
4. Road Infrastructure Anomaly Detection - Speed breakers, potholes, damaged roads
5. Enhanced Explainability Module - Multi-layer visual explanations

Author: AI Project Demonstration
Requirements: ultralytics, opencv-python, torch, numpy

VGG16 is NOT required for this implementation - we use YOLO + Computer Vision techniques
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path


class RoadInfrastructureDetector:
    """Class for detecting road infrastructure anomalies like speed breakers, potholes, damaged roads"""
    
    def __init__(self):
        """Initialize road infrastructure detection parameters"""
        self.speed_breaker_detector = SpeedBreakerDetector()
        self.road_damage_detector = RoadDamageDetector()
    
    def detect_infrastructure_anomalies(self, frame):
        """
        Detect road infrastructure anomalies
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            list: Infrastructure anomaly detections
        """
        anomalies = []
        
        # Detect speed breakers
        speed_breakers = self.speed_breaker_detector.detect(frame)
        anomalies.extend(speed_breakers)
        
        # Detect road damage
        road_damage = self.road_damage_detector.detect(frame)
        anomalies.extend(road_damage)
        
        return anomalies


class SpeedBreakerDetector:
    """Detect speed breakers using computer vision techniques"""
    
    def detect(self, frame):
        """
        Detect speed breakers in the image
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            list: Speed breaker detections
        """
        detections = []
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = frame.shape[:2]
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            
            # Filter based on area (speed breakers are typically medium-sized)
            if 500 < area < 5000:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Speed breakers are typically horizontal and in lower part of image
                aspect_ratio = w / h
                if aspect_ratio > 2.0 and y > height * 0.4:  # Lower half of image
                    
                    # Additional validation using horizontal line detection
                    roi = gray[y:y+h, x:x+w]
                    horizontal_lines = self._detect_horizontal_lines(roi)
                    
                    if horizontal_lines > 0:
                        detections.append({
                            'type': 'speed_breaker',
                            'bbox': [x, y, x+w, y+h],
                            'confidence': min(0.8, 0.3 + (horizontal_lines * 0.1)),
                            'description': 'Speed Breaker Ahead'
                        })
        
        return detections
    
    def _detect_horizontal_lines(self, roi):
        """Detect horizontal lines in ROI (characteristic of speed breakers)"""
        if roi.size == 0:
            return 0
            
        # Apply HoughLines to detect horizontal lines
        lines = cv2.HoughLines(roi, 1, np.pi/180, threshold=30)
        
        horizontal_count = 0
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # Check if line is approximately horizontal (theta close to 0 or pi)
                if abs(theta) < 0.2 or abs(theta - np.pi) < 0.2:
                    horizontal_count += 1
        
        return horizontal_count


class RoadDamageDetector:
    """Detect road damage like potholes, cracks, and surface irregularities"""
    
    def detect(self, frame):
        """
        Detect road damage in the image
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            list: Road damage detections
        """
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect dark spots (potential potholes)
        potholes = self._detect_potholes(gray, frame.shape)
        detections.extend(potholes)
        
        # Detect surface irregularities
        surface_damage = self._detect_surface_damage(gray, frame.shape)
        detections.extend(surface_damage)
        
        return detections
    
    def _detect_potholes(self, gray, frame_shape):
        """Detect potholes (dark circular/oval regions on road)"""
        detections = []
        height, width = frame_shape[:2]
        
        # Apply threshold to find dark regions
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter based on area (potholes are medium-sized)
            if 200 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Potholes are typically in the road area (lower 2/3 of image)
                if y > height * 0.3:
                    # Check circularity (potholes tend to be somewhat circular)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.3:  # Reasonably circular
                            confidence = min(0.7, 0.4 + circularity * 0.3)
                            detections.append({
                                'type': 'pothole',
                                'bbox': [x, y, x+w, y+h],
                                'confidence': confidence,
                                'description': 'Road Damage - Pothole'
                            })
        
        return detections
    
    def _detect_surface_damage(self, gray, frame_shape):
        """Detect surface damage like cracks and rough patches"""
        detections = []
        height, width = frame_shape[:2]
        
        # Use texture analysis to find irregular surfaces
        # Apply Laplacian to detect edges and texture variations
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Threshold to find high-variation areas
        _, thresh = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter based on area
            if 300 < area < 2000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Focus on road surface area
                if y > height * 0.4:
                    detections.append({
                        'type': 'surface_damage',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.5,
                        'description': 'Road Surface Damage'
                    })
        
        return detections


class EnhancedRoadSceneDetector:
    """Enhanced class for road scene anomaly detection system with infrastructure analysis"""
    
    def __init__(self, model_path='yolov8n.pt', input_size=640):
        """
        Initialize the enhanced detection system
        
        Args:
            model_path (str): Path to YOLO model weights
            input_size (int): Input size for YOLO model
        """
        self.input_size = input_size
        self.model = self._load_model(model_path)
        
        # Define classes of interest for road scenes
        self.target_classes = {
            0: 'person',      # pedestrians
            1: 'bicycle',     # bicycles
            2: 'car',         # cars
            3: 'motorcycle',  # motorcycles
            5: 'bus',         # buses
            7: 'truck'        # trucks
        }
        
        # Road infrastructure detection parameters
        self.road_anomaly_detector = RoadInfrastructureDetector()
        
    def _load_model(self, model_path):
        """
        Data Collection Module: Load pre-trained YOLO model
        
        Args:
            model_path (str): Path to model weights
            
        Returns:
            YOLO: Loaded YOLO model
        """
        print(f"Loading YOLO model from: {model_path}")
        try:
            model = YOLO(model_path)
            print("✓ YOLO model loaded successfully")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """
        Preprocessing Module: Resize and normalize input frame
        
        Args:
            frame (np.ndarray): Input frame/image
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        # Resize frame to YOLO input size while maintaining aspect ratio
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        scale = min(self.input_size / width, self.input_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Create padded frame
        padded_frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        
        # Center the resized frame
        y_offset = (self.input_size - new_height) // 2
        x_offset = (self.input_size - new_width) // 2
        padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
        
        return padded_frame, scale, x_offset, y_offset
    
    def detect_objects(self, frame):
        """
        Detection Module: Perform object detection using YOLO
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            list: Detection results with bounding boxes and confidence scores
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection information
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for road scene relevant classes
                    if class_id in self.target_classes and confidence > 0.3:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.target_classes[class_id],
                            'type': 'object'
                        })
        
        return detections
    
    def detect_all_anomalies(self, frame):
        """
        Enhanced Detection Module: Perform both object and infrastructure anomaly detection
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (object_detections, infrastructure_anomalies)
        """
        # Detect vehicles and pedestrians using YOLO
        object_detections = self.detect_objects(frame)
        
        # Detect road infrastructure anomalies
        infrastructure_anomalies = self.road_anomaly_detector.detect_infrastructure_anomalies(frame)
        
        return object_detections, infrastructure_anomalies
    
    def draw_enhanced_explanations(self, frame, object_detections, infrastructure_anomalies):
        """
        Enhanced Explainability Module: Draw both object and infrastructure detections
        
        Args:
            frame (np.ndarray): Input frame
            object_detections (list): List of object detection results
            infrastructure_anomalies (list): List of infrastructure anomaly results
            
        Returns:
            np.ndarray: Frame with drawn explanations
        """
        explained_frame = frame.copy()
        
        # Define colors for different classes
        object_colors = {
            'person': (0, 255, 0),      # Green for pedestrians
            'bicycle': (255, 255, 0),   # Cyan for bicycles
            'car': (255, 0, 0),         # Blue for cars
            'motorcycle': (0, 165, 255), # Orange for motorcycles
            'bus': (255, 0, 255),       # Magenta for buses
            'truck': (0, 0, 255)        # Red for trucks
        }
        
        # Define colors for infrastructure anomalies
        anomaly_colors = {
            'speed_breaker': (0, 255, 255),    # Yellow for speed breakers
            'pothole': (0, 0, 255),            # Red for potholes
            'surface_damage': (0, 128, 255)    # Orange for surface damage
        }
        
        # Draw object detections
        for detection in object_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Get color for this class
            color = object_colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(explained_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                explained_frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                explained_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Draw infrastructure anomaly detections
        for anomaly in infrastructure_anomalies:
            x1, y1, x2, y2 = anomaly['bbox']
            confidence = anomaly['confidence']
            anomaly_type = anomaly['type']
            description = anomaly['description']
            
            # Get color for this anomaly type
            color = anomaly_colors.get(anomaly_type, (128, 128, 128))
            
            # Draw bounding box with dashed line style for anomalies
            self._draw_dashed_rectangle(explained_frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label with description and confidence
            label = f"{description}: {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(
                explained_frame,
                (x1, y2 + 5),
                (x1 + label_width, y2 + label_height + 15),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                explained_frame,
                label,
                (x1, y2 + label_height + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # Add legend
        self._draw_legend(explained_frame, object_colors, anomaly_colors)
        
        return explained_frame
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Draw a dashed rectangle for infrastructure anomalies"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw dashed lines
        dash_length = 10
        
        # Top line
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        
        # Bottom line
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left line
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        
        # Right line
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def _draw_legend(self, frame, object_colors, anomaly_colors):
        """Draw legend explaining the color coding"""
        height, width = frame.shape[:2]
        
        # Legend background
        legend_height = 120
        legend_width = 250
        legend_x = width - legend_width - 10
        legend_y = 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Legend title
        cv2.putText(frame, "Detection Legend", (legend_x + 5, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Object legend
        y_offset = legend_y + 35
        cv2.putText(frame, "Objects:", (legend_x + 5, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 15
        for obj_type, color in list(object_colors.items())[:3]:  # Show first 3
            cv2.rectangle(frame, (legend_x + 5, y_offset - 8), 
                         (legend_x + 20, y_offset + 2), color, -1)
            cv2.putText(frame, obj_type, (legend_x + 25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            y_offset += 12
        
        # Anomaly legend
        y_offset += 5
        cv2.putText(frame, "Road Anomalies:", (legend_x + 5, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 15
        for anomaly_type, color in anomaly_colors.items():
            cv2.rectangle(frame, (legend_x + 5, y_offset - 8), 
                         (legend_x + 20, y_offset + 2), color, -1)
            display_name = anomaly_type.replace('_', ' ').title()
            cv2.putText(frame, display_name, (legend_x + 25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            y_offset += 12
    
    def process_image(self, image_path):
        """
        Process a single image with enhanced detection
        
        Args:
            image_path (str): Path to input image
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"✗ Could not load image: {image_path}")
            return
        
        # Preprocess frame
        processed_frame, scale, x_offset, y_offset = self.preprocess_frame(frame)
        
        # Detect all anomalies
        object_detections, infrastructure_anomalies = self.detect_all_anomalies(processed_frame)
        
        # Adjust bounding boxes back to original frame coordinates
        original_object_detections = self._adjust_coordinates(
            object_detections, scale, x_offset, y_offset, frame.shape
        )
        original_infrastructure_anomalies = self._adjust_coordinates(
            infrastructure_anomalies, scale, x_offset, y_offset, frame.shape
        )
        
        # Draw enhanced explanations
        result_frame = self.draw_enhanced_explanations(
            frame, original_object_detections, original_infrastructure_anomalies
        )
        
        # Display results
        total_detections = len(original_object_detections) + len(original_infrastructure_anomalies)
        print(f"✓ Detected {total_detections} total anomalies:")
        print(f"  - Objects: {len(original_object_detections)}")
        print(f"  - Infrastructure: {len(original_infrastructure_anomalies)}")
        
        for detection in original_object_detections:
            print(f"    • {detection['class_name']}: {detection['confidence']:.3f}")
        
        for anomaly in original_infrastructure_anomalies:
            print(f"    • {anomaly['description']}: {anomaly['confidence']:.3f}")
        
        # Show result
        cv2.imshow('Enhanced Road Scene Detection', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _adjust_coordinates(self, detections, scale, x_offset, y_offset, frame_shape):
        """Adjust detection coordinates back to original frame"""
        adjusted_detections = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Adjust coordinates back to original frame
            x1 = int((x1 - x_offset) / scale)
            y1 = int((y1 - y_offset) / scale)
            x2 = int((x2 - x_offset) / scale)
            y2 = int((y2 - y_offset) / scale)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame_shape[1]))
            y1 = max(0, min(y1, frame_shape[0]))
            x2 = max(0, min(x2, frame_shape[1]))
            y2 = max(0, min(y2, frame_shape[0]))
            
            adjusted_detection = detection.copy()
            adjusted_detection['bbox'] = [x1, y1, x2, y2]
            adjusted_detections.append(adjusted_detection)
        
        return adjusted_detections


def main():
    """Main function to run the enhanced detection system"""
    parser = argparse.ArgumentParser(description='Enhanced Road Scene Anomaly Detection Demo')
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to input image or video file')
    parser.add_argument('--model', '-m', default='yolov8n.pt',
                       help='Path to YOLO model weights (default: yolov8n.pt)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"✗ Input file not found: {args.input}")
        return
    
    # Initialize enhanced detector
    try:
        detector = EnhancedRoadSceneDetector(model_path=args.model)
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        return
    
    # Determine file type and process accordingly
    file_extension = Path(args.input).suffix.lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # Process image
        detector.process_image(args.input)
    else:
        print(f"✗ Unsupported file format: {file_extension}")
        print("Supported formats: Images (.jpg, .png, .bmp)")


if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Road Scene Anomaly Detection System - Demo")
    print("=" * 70)
    print("Modules implemented:")
    print("✓ Data Collection Module")
    print("✓ Preprocessing Module") 
    print("✓ Detection Module (YOLO)")
    print("✓ Road Infrastructure Anomaly Detection")
    print("✓ Enhanced Explainability Module")
    print("=" * 70)
    print("🚫 VGG16 is NOT required - Using YOLO + Computer Vision")
    print("=" * 70)
    
    main()