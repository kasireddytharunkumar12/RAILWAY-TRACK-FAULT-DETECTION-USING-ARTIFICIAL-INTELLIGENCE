"""
Data Preprocessing and Augmentation Pipeline

This module provides comprehensive data preprocessing and augmentation
capabilities for railway track fault detection, including image
normalization, geometric transformations, and domain-specific augmentations.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
from pathlib import Path
import json
import random
from PIL import Image
import matplotlib.pyplot as plt


class RailwayImagePreprocessor:
    """
    Comprehensive image preprocessing pipeline for railway fault detection.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (640, 640),
                 normalize: bool = True,
                 enhance_contrast: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
            enhance_contrast: Whether to apply contrast enhancement
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        
        # Normalization parameters (ImageNet statistics)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def resize_image(self, image: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            # Calculate scaling factor
            h, w = image.shape[:2]
            target_h, target_w = self.target_size
            
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Create padded image
            padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
            
            # Calculate padding offsets
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place resized image in center
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            return padded
        else:
            return cv2.resize(image, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better fault detection.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        enhanced = image.copy()
        
        if self.enhance_contrast:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image (0-255 range)
            
        Returns:
            Normalized image
        """
        if self.normalize:
            # Convert to float and normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            for i in range(3):
                normalized[:, :, i] = (normalized[:, :, i] - self.mean[i]) / self.std[i]
            
            return normalized
        
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            image: Input image (path or numpy array)
            
        Returns:
            Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        resized = self.resize_image(image)
        
        # Enhance quality
        enhanced = self.enhance_image_quality(resized)
        
        # Normalize
        normalized = self.normalize_image(enhanced)
        
        return normalized


class RailwayAugmentationPipeline:
    """
    Advanced augmentation pipeline specifically designed for railway images.
    """
    
    def __init__(self, 
                 mode: str = 'train',
                 augmentation_prob: float = 0.8,
                 target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize augmentation pipeline.
        
        Args:
            mode: 'train', 'val', or 'test'
            augmentation_prob: Probability of applying augmentations
            target_size: Target image size
        """
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.target_size = target_size
        
        self.train_transforms = self._create_train_transforms()
        self.val_transforms = self._create_val_transforms()
    
    def _create_train_transforms(self) -> A.Compose:
        """Create training augmentation pipeline."""
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=10,
                p=0.3
            ),
            
            # Perspective and distortion
            A.Perspective(scale=(0.05, 0.1), p=0.2),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.2
            ),
            
            # Color and lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Weather and environmental effects
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=7,
                brightness_coefficient=0.7,
                rain_type=None,
                p=0.1
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=6,
                num_flare_circles_upper=10,
                src_radius=160,
                src_color=(255, 255, 255),
                p=0.1
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.Blur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Resize and normalize
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _create_val_transforms(self) -> A.Compose:
        """Create validation/test augmentation pipeline."""
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Apply augmentations to image and mask.
        
        Args:
            image: Input image
            mask: Optional segmentation mask
            
        Returns:
            Dictionary with augmented image and mask
        """
        if self.mode == 'train':
            transforms = self.train_transforms
        else:
            transforms = self.val_transforms
        
        if mask is not None:
            result = transforms(image=image, mask=mask)
            return {
                'image': result['image'],
                'mask': result['mask']
            }
        else:
            result = transforms(image=image)
            return {
                'image': result['image']
            }


class RailwayFaultDataset(Dataset):
    """
    PyTorch Dataset for railway fault detection.
    """
    
    def __init__(self,
                 data_dir: str,
                 annotations_file: str,
                 mode: str = 'train',
                 transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            annotations_file: Path to annotations JSON file
            mode: 'train', 'val', or 'test'
            transform: Optional transform function
            target_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create class mapping
        self.class_names = [
            'rail_crack', 'rail_break', 'fastener_loose',
            'fastener_missing', 'track_misalign', 'normal'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Set up transforms
        if transform is None:
            self.transform = RailwayAugmentationPipeline(
                mode=mode,
                target_size=target_size
            )
        else:
            self.transform = transform
        
        # Filter annotations by mode
        self.samples = [
            ann for ann in self.annotations
            if ann.get('split', 'train') == mode
        ]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image and label tensors
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_dir / sample['image_path']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label
        label = self.class_to_idx[sample['class']]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image)
            image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': str(image_path),
            'class_name': sample['class']
        }


def create_data_loaders(train_dir: str,
                       val_dir: str,
                       train_annotations: str,
                       val_annotations: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       target_size: Tuple[int, int] = (640, 640)) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_dir: Training images directory
        val_dir: Validation images directory
        train_annotations: Training annotations file
        val_annotations: Validation annotations file
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        target_size: Target image size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = RailwayFaultDataset(
        data_dir=train_dir,
        annotations_file=train_annotations,
        mode='train',
        target_size=target_size
    )
    
    val_dataset = RailwayFaultDataset(
        data_dir=val_dir,
        annotations_file=val_annotations,
        mode='val',
        target_size=target_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def visualize_augmentations(dataset: RailwayFaultDataset, 
                          num_samples: int = 4,
                          save_path: Optional[str] = None):
    """
    Visualize dataset samples with augmentations.
    
    Args:
        dataset: Railway fault dataset
        num_samples: Number of samples to visualize
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(num_samples):
        sample = dataset[i]
        image = sample['image']
        class_name = sample['class_name']
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            image = image.permute(1, 2, 0).numpy()
        
        # Original image (top row)
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Sample {i+1}: {class_name}')
        axes[0, i].axis('off')
        
        # Augmented version (bottom row)
        sample_aug = dataset[i]  # Get another augmented version
        image_aug = sample_aug['image']
        
        if isinstance(image_aug, torch.Tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_aug = image_aug * std + mean
            image_aug = torch.clamp(image_aug, 0, 1)
            image_aug = image_aug.permute(1, 2, 0).numpy()
        
        axes[1, i].imshow(image_aug)
        axes[1, i].set_title(f'Augmented {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Railway Data Preprocessing Pipeline")
    
    # Test preprocessor
    preprocessor = RailwayImagePreprocessor(target_size=(640, 640))
    
    # Test augmentation pipeline
    aug_pipeline = RailwayAugmentationPipeline(mode='train')
    
    # Create dummy image for testing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test preprocessing
    processed = preprocessor.preprocess(dummy_image)
    print(f"Original shape: {dummy_image.shape}")
    print(f"Processed shape: {processed.shape}")
    
    # Test augmentation
    augmented = aug_pipeline(dummy_image)
    print(f"Augmented image shape: {augmented['image'].shape}")
    
    print("Data preprocessing pipeline initialized successfully!")

