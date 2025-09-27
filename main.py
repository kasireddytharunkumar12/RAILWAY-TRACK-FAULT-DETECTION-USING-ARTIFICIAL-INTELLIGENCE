"""
Railway Track Fault Detection System - Main Application

This is the main entry point for the railway track fault detection system.
It provides a comprehensive interface for training models, running inference,
and analyzing railway infrastructure for potential faults.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_detector import RailwayFaultCNN, FaultDetectionTrainer, create_model
from models.yolo_detector import RailwayYOLODetector, create_yolo_detector
from data.preprocessing import (
    RailwayImagePreprocessor, 
    RailwayAugmentationPipeline,
    RailwayFaultDataset,
    create_data_loaders
)
from utils.visualization import ResultVisualizer
from utils.metrics import ModelEvaluator


class RailwayFaultDetectionSystem:
    """
    Main system class that orchestrates all components for railway fault detection.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the railway fault detection system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.cnn_model = None
        self.yolo_detector = None
        self.preprocessor = None
        self.evaluator = None
        self.visualizer = None
        
        self.logger.info(f"System initialized on device: {self.device}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load system configuration."""
        default_config = {
            'model': {
                'cnn': {
                    'num_classes': 6,
                    'pretrained': True,
                    'model_path': None
                },
                'yolo': {
                    'model_path': None,
                    'confidence_threshold': 0.5,
                    'iou_threshold': 0.45
                }
            },
            'data': {
                'target_size': [640, 640],
                'batch_size': 32,
                'num_workers': 4
            },
            'training': {
                'epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5
            },
            'paths': {
                'models_dir': 'models',
                'results_dir': 'results',
                'logs_dir': 'logs'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configurations
            default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['paths']['logs_dir'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'railway_detection.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_models(self):
        """Initialize AI models for fault detection."""
        self.logger.info("Initializing AI models...")
        
        # Initialize CNN model
        cnn_config = self.config['model']['cnn']
        self.cnn_model = create_model(
            num_classes=cnn_config['num_classes'],
            pretrained=cnn_config['pretrained']
        ).to(self.device)
        
        # Load pretrained weights if available
        if cnn_config['model_path'] and os.path.exists(cnn_config['model_path']):
            self.cnn_model.load_state_dict(torch.load(cnn_config['model_path']))
            self.logger.info(f"Loaded CNN model from {cnn_config['model_path']}")
        
        # Initialize YOLO detector
        yolo_config = self.config['model']['yolo']
        self.yolo_detector = create_yolo_detector(
            model_path=yolo_config['model_path'],
            device=str(self.device)
        )
        
        # Set YOLO thresholds
        self.yolo_detector.set_thresholds(
            confidence=yolo_config['confidence_threshold'],
            iou=yolo_config['iou_threshold']
        )
        
        # Initialize preprocessor
        data_config = self.config['data']
        self.preprocessor = RailwayImagePreprocessor(
            target_size=tuple(data_config['target_size'])
        )
        
        # Initialize evaluator and visualizer
        self.evaluator = ModelEvaluator()
        self.visualizer = ResultVisualizer()
        
        self.logger.info("Models initialized successfully")
    
    def train_cnn_model(self, 
                       train_dir: str, 
                       val_dir: str,
                       train_annotations: str,
                       val_annotations: str) -> Dict:
        """
        Train the CNN model for fault classification.
        
        Args:
            train_dir: Training images directory
            val_dir: Validation images directory
            train_annotations: Training annotations file
            val_annotations: Validation annotations file
            
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting CNN model training...")
        
        if self.cnn_model is None:
            self.initialize_models()
        
        # Create data loaders
        data_config = self.config['data']
        train_loader, val_loader = create_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            target_size=tuple(data_config['target_size'])
        )
        
        # Initialize trainer
        trainer = FaultDetectionTrainer(self.cnn_model, str(self.device))
        
        # Train model
        training_config = self.config['training']
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config['epochs']
        )
        
        # Save trained model
        models_dir = Path(self.config['paths']['models_dir'])
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / 'cnn_fault_detector.pth'
        torch.save(self.cnn_model.state_dict(), model_path)
        
        self.logger.info(f"CNN model training completed. Model saved to {model_path}")
        
        return history
    
    def train_yolo_model(self, 
                        data_config_path: str,
                        epochs: int = 100) -> Dict:
        """
        Train the YOLO model for object detection.
        
        Args:
            data_config_path: Path to YOLO dataset configuration
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        self.logger.info("Starting YOLO model training...")
        
        if self.yolo_detector is None:
            self.initialize_models()
        
        # Train YOLO model
        results = self.yolo_detector.train(
            data_config=data_config_path,
            epochs=epochs,
            imgsz=self.config['data']['target_size'][0],
            batch_size=self.config['data']['batch_size']
        )
        
        self.logger.info("YOLO model training completed")
        
        return results
    
    def detect_faults_single_image(self, 
                                  image_path: str,
                                  use_cnn: bool = True,
                                  use_yolo: bool = True) -> Dict:
        """
        Detect faults in a single image using both models.
        
        Args:
            image_path: Path to input image
            use_cnn: Whether to use CNN classifier
            use_yolo: Whether to use YOLO detector
            
        Returns:
            Detection results dictionary
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {
            'image_path': image_path,
            'image_shape': image.shape,
            'cnn_results': None,
            'yolo_results': None,
            'combined_analysis': None
        }
        
        # CNN Classification
        if use_cnn and self.cnn_model is not None:
            processed_image = self.preprocessor.preprocess(image_rgb)
            input_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            cnn_predictions = self.cnn_model.predict(input_tensor)
            results['cnn_results'] = {
                'prediction': cnn_predictions['predictions'].cpu().numpy()[0],
                'confidence': cnn_predictions['confidence'].cpu().numpy()[0],
                'probabilities': cnn_predictions['probabilities'].cpu().numpy()[0]
            }
        
        # YOLO Object Detection
        if use_yolo and self.yolo_detector is not None:
            yolo_results = self.yolo_detector.detect(image_rgb)
            results['yolo_results'] = yolo_results
            
            # Analyze fault severity
            severity_analysis = self.yolo_detector.analyze_fault_severity(
                yolo_results['detections']
            )
            results['yolo_results']['severity_analysis'] = severity_analysis
        
        # Combined analysis
        results['combined_analysis'] = self._combine_detection_results(
            results['cnn_results'],
            results['yolo_results']
        )
        
        return results
    
    def detect_faults_batch(self, 
                           image_paths: List[str],
                           output_dir: str,
                           use_cnn: bool = True,
                           use_yolo: bool = True) -> List[Dict]:
        """
        Detect faults in multiple images.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            use_cnn: Whether to use CNN classifier
            use_yolo: Whether to use YOLO detector
            
        Returns:
            List of detection results
        """
        self.logger.info(f"Processing {len(image_paths)} images...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                # Detect faults
                result = self.detect_faults_single_image(
                    image_path=image_path,
                    use_cnn=use_cnn,
                    use_yolo=use_yolo
                )
                
                # Visualize results
                if result['yolo_results'] and result['yolo_results']['detections']:
                    vis_image = self._visualize_detection_result(image_path, result)
                    
                    # Save visualization
                    output_file = output_path / f"result_{i:04d}.jpg"
                    cv2.imwrite(str(output_file), vis_image)
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Save batch results
        results_file = output_path / "batch_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Batch processing completed. Results saved to {output_dir}")
        
        return results
    
    def _combine_detection_results(self, 
                                  cnn_results: Optional[Dict],
                                  yolo_results: Optional[Dict]) -> Dict:
        """Combine CNN and YOLO detection results."""
        combined = {
            'overall_confidence': 0.0,
            'fault_detected': False,
            'primary_fault_type': 'normal',
            'risk_level': 'low',
            'recommendations': []
        }
        
        if cnn_results:
            cnn_confidence = float(cnn_results['confidence'])
            cnn_prediction = int(cnn_results['prediction'])
            
            class_names = [
                'rail_crack', 'rail_break', 'fastener_loose',
                'fastener_missing', 'track_misalign', 'normal'
            ]
            
            if cnn_prediction < len(class_names):
                predicted_class = class_names[cnn_prediction]
                if predicted_class != 'normal' and cnn_confidence > 0.7:
                    combined['fault_detected'] = True
                    combined['primary_fault_type'] = predicted_class
                    combined['overall_confidence'] = cnn_confidence
        
        if yolo_results and yolo_results.get('severity_analysis'):
            severity = yolo_results['severity_analysis']
            
            if severity['requires_immediate_attention']:
                combined['fault_detected'] = True
                combined['risk_level'] = 'high'
                combined['recommendations'].append('Immediate inspection required')
            
            if severity['total_faults'] > 0:
                combined['fault_detected'] = True
                combined['overall_confidence'] = max(
                    combined['overall_confidence'], 0.8
                )
        
        # Generate recommendations
        if combined['fault_detected']:
            if combined['risk_level'] == 'high':
                combined['recommendations'].extend([
                    'Stop train operations immediately',
                    'Deploy maintenance crew',
                    'Conduct detailed inspection'
                ])
            else:
                combined['recommendations'].extend([
                    'Schedule maintenance inspection',
                    'Monitor closely during operations',
                    'Document fault location'
                ])
        
        return combined
    
    def _visualize_detection_result(self, image_path: str, result: Dict) -> np.ndarray:
        """Visualize detection results on image."""
        image = cv2.imread(image_path)
        
        if result['yolo_results'] and result['yolo_results']['detections']:
            detections = result['yolo_results']['detections']
            image = self.yolo_detector.visualize_detections(image, detections)
        
        return image
    
    def evaluate_model_performance(self, 
                                 test_dir: str,
                                 test_annotations: str) -> Dict:
        """
        Evaluate model performance on test dataset.
        
        Args:
            test_dir: Test images directory
            test_annotations: Test annotations file
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating model performance...")
        
        if self.evaluator is None:
            self.evaluator = ModelEvaluator()
        
        # Create test dataset
        test_dataset = RailwayFaultDataset(
            data_dir=test_dir,
            annotations_file=test_annotations,
            mode='test',
            target_size=tuple(self.config['data']['target_size'])
        )
        
        # Evaluate models
        metrics = self.evaluator.evaluate_models(
            cnn_model=self.cnn_model,
            yolo_detector=self.yolo_detector,
            test_dataset=test_dataset,
            device=self.device
        )
        
        # Save evaluation results
        results_dir = Path(self.config['paths']['results_dir'])
        results_dir.mkdir(exist_ok=True)
        
        metrics_file = results_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation completed. Metrics saved to {metrics_file}")
        
        return metrics


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Railway Track Fault Detection System"
    )
    
    parser.add_argument(
        '--mode',
        choices=['train_cnn', 'train_yolo', 'detect', 'evaluate'],
        required=True,
        help='Operation mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image for detection'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        help='Directory containing images for batch processing'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = RailwayFaultDetectionSystem(config_path=args.config)
    system.initialize_models()
    
    if args.mode == 'detect':
        if args.image:
            # Single image detection
            result = system.detect_faults_single_image(args.image)
            print(json.dumps(result, indent=2, default=str))
        
        elif args.images_dir:
            # Batch detection
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(Path(args.images_dir).glob(ext))
            
            results = system.detect_faults_batch(
                image_paths=[str(p) for p in image_paths],
                output_dir=args.output_dir
            )
            
            print(f"Processed {len(results)} images. Results saved to {args.output_dir}")
    
    else:
        print(f"Mode '{args.mode}' requires additional implementation for CLI interface")


if __name__ == "__main__":
    main()

