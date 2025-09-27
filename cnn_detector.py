"""
CNN-based Railway Track Fault Detection Model

This module implements a Convolutional Neural Network for detecting
various types of faults in railway track infrastructure including
cracks, broken rails, and fastener defects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import numpy as np


class RailwayFaultCNN(nn.Module):
    """
    CNN model for railway track fault detection and classification.
    
    Architecture:
    - ResNet50 backbone for feature extraction
    - Custom classification head for fault types
    - Multi-scale feature processing for different fault sizes
    """
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of fault classes to detect
                        (rail_crack, rail_break, fastener_loose, 
                         fastener_missing, track_misalign, normal)
            pretrained: Whether to use pretrained ResNet weights
        """
        super(RailwayFaultCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Backbone: ResNet50 with modifications
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Feature dimensions from ResNet50
        self.feature_dim = 2048
        
        # Multi-scale feature processing
        self.feature_pyramid = self._build_feature_pyramid()
        
        # Classification head
        self.classifier = self._build_classifier()
        
        # Attention mechanism for critical region focus
        self.attention = self._build_attention_module()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def _build_feature_pyramid(self) -> nn.ModuleDict:
        """Build feature pyramid network for multi-scale processing."""
        return nn.ModuleDict({
            'conv1': nn.Conv2d(self.feature_dim, 256, 1),
            'conv2': nn.Conv2d(256, 256, 3, padding=1),
            'conv3': nn.Conv2d(256, 256, 3, padding=1),
        })
    
    def _build_classifier(self) -> nn.Sequential:
        """Build classification head."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
    
    def _build_attention_module(self) -> nn.Module:
        """Build attention mechanism for focusing on critical regions."""
        return nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - features: Extracted features
                - attention_map: Attention weights
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Reshape features for spatial processing
        batch_size = features.size(0)
        features = features.view(batch_size, self.feature_dim, 1, 1)
        
        # Process through feature pyramid
        fpn_features = self.feature_pyramid['conv1'](features)
        fpn_features = F.relu(fpn_features)
        fpn_features = self.feature_pyramid['conv2'](fpn_features)
        fpn_features = F.relu(fpn_features)
        fpn_features = self.feature_pyramid['conv3'](fpn_features)
        
        # Apply attention mechanism
        attention_weights = self.attention(fpn_features)
        attended_features = fpn_features * attention_weights
        
        # Classification
        logits = self.classifier(attended_features)
        
        return {
            'logits': logits,
            'features': fpn_features,
            'attention_map': attention_weights
        }
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input tensor
            threshold: Confidence threshold for positive predictions
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs['logits'], dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': confidence,
                'attention_map': outputs['attention_map']
            }


class FaultDetectionTrainer:
    """
    Training class for the railway fault detection model.
    """
    
    def __init__(self, model: RailwayFaultCNN, device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: The CNN model to train
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5, 
            factor=0.5
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs['logits'], targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs['logits'], targets)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 100) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if patience_counter >= max_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }


def create_model(num_classes: int = 6, pretrained: bool = True) -> RailwayFaultCNN:
    """
    Factory function to create a railway fault detection model.
    
    Args:
        num_classes: Number of fault classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Initialized model
    """
    return RailwayFaultCNN(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Example usage
    model = create_model(num_classes=6)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    outputs = model(dummy_input)
    
    print(f"Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Attention map shape: {outputs['attention_map'].shape}")

