"""
Visualization Utilities for Railway Fault Detection

This module provides comprehensive visualization capabilities for
displaying detection results, training metrics, and analysis reports.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class ResultVisualizer:
    """
    Comprehensive visualization class for railway fault detection results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {
            'rail_crack': '#FF0000',      # Red
            'rail_break': '#FF8C00',      # Dark Orange
            'fastener_loose': '#FFD700',  # Gold
            'fastener_missing': '#FF1493', # Deep Pink
            'track_misalign': '#00FFFF',  # Cyan
            'normal': '#00FF00'           # Green
        }
    
    def plot_training_history(self, 
                            history: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot training and validation metrics over epochs.
        
        Args:
            history: Training history dictionary
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Training and validation loss
        axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training and validation accuracy
        axes[0, 1].plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Loss difference (overfitting indicator)
        loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
        axes[1, 0].plot(epochs, loss_diff, 'g-', label='Val Loss - Train Loss')
        axes[1, 0].set_title('Overfitting Indicator')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Learning rate (if available)
        if 'learning_rates' in history:
            axes[1, 1].plot(epochs, history['learning_rates'], 'purple', label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            class_names: List[str],
                            normalize: bool = True,
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix with annotations.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize the matrix
            save_path: Optional path to save the plot
        """
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt=fmt, 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_detection_results(self, 
                             image: np.ndarray,
                             detections: List[Dict],
                             title: str = "Fault Detection Results",
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detection results on an image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Convert hex to BGR for OpenCV
            hex_color = color.lstrip("#")
            color_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color_bgr,
                -1
            )
            
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Display with matplotlib
        plt.figure(figsize=self.figsize)
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return annotated_image
    
    def plot_performance_metrics(self, 
                               metrics: Dict[str, float],
                               save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics as bar chart.
        
        Args:
            metrics: Dictionary of metric names and values
            save_path: Optional path to save the plot
        """
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metric_names, metric_values, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Performance Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_distribution(self, 
                              class_counts: Dict[str, int],
                              save_path: Optional[str] = None) -> None:
        """
        Plot class distribution as pie chart and bar chart.
        
        Args:
            class_counts: Dictionary of class names and counts
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = [self.colors.get(cls, '#888888') for cls in classes]
        
        # Pie chart
        ax1.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Class Distribution (Pie Chart)')
        
        # Bar chart
        bars = ax2.bar(classes, counts, color=colors)
        ax2.set_title('Class Distribution (Bar Chart)')
        ax2.set_xlabel('Fault Classes')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, 
                                   results: List[Dict],
                                   save_path: Optional[str] = None) -> None:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            results: List of detection results
            save_path: Optional path to save HTML file
        """
        # Prepare data
        fault_counts = {}
        confidence_scores = []
        severity_levels = []
        
        for result in results:
            if result.get('yolo_results') and result['yolo_results'].get('detections'):
                for detection in result['yolo_results']['detections']:
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    fault_counts[class_name] = fault_counts.get(class_name, 0) + 1
                    confidence_scores.append(confidence)
                    
                    # Determine severity based on class
                    if class_name in ['rail_break', 'track_misalign']:
                        severity_levels.append('Critical')
                    elif class_name in ['rail_crack', 'fastener_missing']:
                        severity_levels.append('High')
                    elif class_name == 'fastener_loose':
                        severity_levels.append('Medium')
                    else:
                        severity_levels.append('Low')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fault Distribution', 'Confidence Scores', 
                          'Severity Analysis', 'Detection Timeline'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Fault distribution pie chart
        if fault_counts:
            fig.add_trace(
                go.Pie(labels=list(fault_counts.keys()), 
                      values=list(fault_counts.values()),
                      name="Fault Distribution"),
                row=1, col=1
            )
        
        # Confidence scores histogram
        if confidence_scores:
            fig.add_trace(
                go.Histogram(x=confidence_scores, 
                           name="Confidence Scores",
                           nbinsx=20),
                row=1, col=2
            )
        
        # Severity analysis
        if severity_levels:
            severity_counts = pd.Series(severity_levels).value_counts()
            fig.add_trace(
                go.Bar(x=severity_counts.index, 
                      y=severity_counts.values,
                      name="Severity Levels"),
                row=2, col=1
            )
        
        # Detection timeline (placeholder)
        fig.add_trace(
            go.Scatter(x=list(range(len(results))), 
                      y=[len(r.get('yolo_results', {}).get('detections', [])) 
                         for r in results],
                      mode='lines+markers',
                      name="Detections per Image"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Railway Fault Detection Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_attention_maps(self, 
                          image: np.ndarray,
                          attention_map: np.ndarray,
                          save_path: Optional[str] = None) -> None:
        """
        Visualize CNN attention maps.
        
        Args:
            image: Original image
            attention_map: Attention weights from CNN
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        attention_resized = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        im1 = axes[1].imshow(attention_resized, cmap='hot', alpha=0.8)
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        overlay = image.copy()
        attention_colored = plt.cm.hot(attention_resized)[:, :, :3]
        overlay = cv2.addWeighted(overlay.astype(np.float32), 0.6, 
                                attention_colored.astype(np.float32), 0.4, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_summary_report(results: List[Dict], 
                        output_path: str) -> None:
    """
    Create a comprehensive summary report.
    
    Args:
        results: List of detection results
        output_path: Path to save the report
    """
    # Calculate summary statistics
    total_images = len(results)
    total_detections = sum(len(r.get('yolo_results', {}).get('detections', [])) 
                          for r in results)
    
    fault_counts = {}
    high_risk_images = 0
    
    for result in results:
        if result.get('combined_analysis'):
            analysis = result['combined_analysis']
            if analysis['fault_detected']:
                fault_type = analysis['primary_fault_type']
                fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
                
                if analysis['risk_level'] == 'high':
                    high_risk_images += 1
    
    # Create report
    report = f"""
    Railway Track Fault Detection Summary Report
    ==========================================
    
    Analysis Overview:
    - Total Images Processed: {total_images}
    - Total Detections Found: {total_detections}
    - Images with Faults: {len(fault_counts)}
    - High Risk Images: {high_risk_images}
    
    Fault Distribution:
    """
    
    for fault_type, count in fault_counts.items():
        percentage = (count / total_images) * 100
        report += f"    - {fault_type}: {count} ({percentage:.1f}%)\n"
    
    report += f"""
    
    Risk Assessment:
    - High Risk: {high_risk_images} images ({(high_risk_images/total_images)*100:.1f}%)
    - Medium/Low Risk: {total_images - high_risk_images} images
    
    Recommendations:
    - Immediate attention required for {high_risk_images} locations
    - Schedule maintenance for {len(fault_counts)} fault locations
    - Continue monitoring for remaining {total_images - len(fault_counts)} locations
    """
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    visualizer = ResultVisualizer()
    
    # Test with dummy data
    dummy_history = {
        'train_losses': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_losses': [0.9, 0.7, 0.5, 0.4, 0.3],
        'train_accuracies': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracies': [0.55, 0.65, 0.75, 0.8, 0.85]
    }
    
    print("Visualization utilities initialized successfully!")
    print("Available methods:")
    print("- plot_training_history()")
    print("- plot_confusion_matrix()")
    print("- plot_detection_results()")
    print("- plot_performance_metrics()")
    print("- create_interactive_dashboard()")

