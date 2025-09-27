"""
YOLO-based Railway Track Fault Detection Model

This module implements a YOLO (You Only Look Once) model for real-time
detection and localization of railway track faults including cracks,
broken components, and fastener issues.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional, Union
import yaml
from pathlib import Path


class RailwayYOLODetector:
    """
    YOLO-based detector for railway track faults with real-time capabilities.
    
    This class wraps the YOLOv8 model and provides railway-specific
    functionality for fault detection and localization.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to trained YOLO model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.class_names = [
            'rail_crack', 'rail_break', 'fastener_loose',
            'fastener_missing', 'track_misalign', 'normal'
        ]
        self.class_colors = [
            (255, 0, 0),    # Red for rail_crack
            (255, 165, 0),  # Orange for rail_break
            (255, 255, 0),  # Yellow for fastener_loose
            (255, 0, 255),  # Magenta for fastener_missing
            (0, 255, 255),  # Cyan for track_misalign
            (0, 255, 0),    # Green for normal
        ]
        
        # Initialize YOLO model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Start with YOLOv8 pretrained model
            self.model = YOLO('yolov8n.pt')
        
        # Configuration
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 100
    
    def train(self, 
              data_config: str, 
              epochs: int = 100, 
              imgsz: int = 640,
              batch_size: int = 16,
              **kwargs) -> Dict:
        """
        Train the YOLO model on railway fault detection dataset.
        
        Args:
            data_config: Path to dataset configuration YAML file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Training batch size
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            **kwargs
        )
        
        return results
    
    def detect(self, 
               image: Union[str, np.ndarray], 
               conf_threshold: Optional[float] = None,
               iou_threshold: Optional[float] = None) -> Dict:
        """
        Detect faults in a single image.
        
        Args:
            image: Input image (path or numpy array)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary containing detection results
        """
        # Set thresholds
        conf = conf_threshold or self.confidence_threshold
        iou = iou_threshold or self.iou_threshold
        
        # Run inference
        results = self.model(
            image,
            conf=conf,
            iou=iou,
            max_det=self.max_detections,
            device=self.device
        )
        
        # Process results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.class_names[class_ids[i]],
                        'color': self.class_colors[class_ids[i]]
                    }
                    detections.append(detection)
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_shape': image.shape if isinstance(image, np.ndarray) else None
        }
    
    def detect_batch(self, 
                     images: List[Union[str, np.ndarray]],
                     conf_threshold: Optional[float] = None,
                     iou_threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect faults in multiple images.
        
        Args:
            images: List of input images
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detection results for each image
        """
        results = []
        for image in images:
            result = self.detect(image, conf_threshold, iou_threshold)
            results.append(result)
        
        return results
    
    def visualize_detections(self, 
                           image: np.ndarray, 
                           detections: List[Dict],
                           show_confidence: bool = True,
                           thickness: int = 2) -> np.ndarray:
        """
        Visualize detections on an image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            show_confidence: Whether to show confidence scores
            thickness: Line thickness for bounding boxes
            
        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = detection['color']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}"
            if show_confidence:
                label += f": {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_height - baseline),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return vis_image
    
    def analyze_fault_severity(self, detections: List[Dict]) -> Dict:
        """
        Analyze the severity of detected faults.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Fault severity analysis
        """
        severity_map = {
            'rail_crack': 'high',
            'rail_break': 'critical',
            'fastener_loose': 'medium',
            'fastener_missing': 'high',
            'track_misalign': 'high',
            'normal': 'none'
        }
        
        fault_counts = {}
        max_severity = 'none'
        severity_levels = ['none', 'low', 'medium', 'high', 'critical']
        
        for detection in detections:
            class_name = detection['class_name']
            severity = severity_map.get(class_name, 'unknown')
            
            # Count faults by type
            if class_name not in fault_counts:
                fault_counts[class_name] = 0
            fault_counts[class_name] += 1
            
            # Track maximum severity
            if severity in severity_levels:
                current_level = severity_levels.index(severity)
                max_level = severity_levels.index(max_severity)
                if current_level > max_level:
                    max_severity = severity
        
        return {
            'max_severity': max_severity,
            'fault_counts': fault_counts,
            'total_faults': len([d for d in detections if d['class_name'] != 'normal']),
            'requires_immediate_attention': max_severity in ['high', 'critical']
        }
    
    def export_model(self, format: str = 'onnx', **kwargs) -> str:
        """
        Export the model to different formats for deployment.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            **kwargs: Additional export parameters
            
        Returns:
            Path to exported model
        """
        return self.model.export(format=format, **kwargs)
    
    def set_thresholds(self, confidence: float, iou: float):
        """
        Set detection thresholds.
        
        Args:
            confidence: Confidence threshold (0.0 - 1.0)
            iou: IoU threshold for NMS (0.0 - 1.0)
        """
        self.confidence_threshold = confidence
        self.iou_threshold = iou


class RailwayDatasetConfig:
    """
    Helper class to create YOLO dataset configuration.
    """
    
    @staticmethod
    def create_config(train_path: str, 
                     val_path: str, 
                     test_path: str = None,
                     class_names: List[str] = None) -> str:
        """
        Create YOLO dataset configuration file.
        
        Args:
            train_path: Path to training images
            val_path: Path to validation images
            test_path: Path to test images (optional)
            class_names: List of class names
            
        Returns:
            Path to created configuration file
        """
        if class_names is None:
            class_names = [
                'rail_crack', 'rail_break', 'fastener_loose',
                'fastener_missing', 'track_misalign', 'normal'
            ]
        
        config = {
            'train': train_path,
            'val': val_path,
            'nc': len(class_names),
            'names': class_names
        }
        
        if test_path:
            config['test'] = test_path
        
        config_path = 'railway_dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path


def create_yolo_detector(model_path: Optional[str] = None, 
                        device: str = 'cuda') -> RailwayYOLODetector:
    """
    Factory function to create a YOLO detector.
    
    Args:
        model_path: Path to trained model weights
        device: Device for inference
        
    Returns:
        Initialized YOLO detector
    """
    return RailwayYOLODetector(model_path=model_path, device=device)


if __name__ == "__main__":
    # Example usage
    detector = create_yolo_detector()
    
    print("YOLO Railway Fault Detector initialized successfully!")
    print(f"Class names: {detector.class_names}")
    print(f"Device: {detector.device}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print(f"IoU threshold: {detector.iou_threshold}")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = detector.detect(dummy_image)
    print(f"Test detection completed. Found {results['num_detections']} objects.")

