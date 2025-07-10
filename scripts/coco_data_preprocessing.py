"""
Car Damage Analysis - COCO Data Preprocessing
===========================================

This script processes COCO-format annotations and converts them to YOLO format
for yolo11n training. Handles detection, segmentation, and severity estimation.

Features:
- COCO to YOLO conversion
- Multi-modal data processing (Image, Edge, Mask)
- Severity estimation from mask areas
- Data augmentation and validation
- Comprehensive dataset analysis

Author: AI Engineer
Date: July 2025
"""

import os
import json
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarDamageCOCOProcessor:
    """
    Advanced COCO data processor for car damage analysis
    """
    
    def __init__(self, data_dir="data/CarDD_COCO", output_dir="yolo_dataset"):
        """
        Initialize the COCO processor
        
        Args:
            data_dir (str): Directory containing COCO data
            output_dir (str): Output directory for YOLO dataset
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Data paths - CarDD specific structure
        self.metadata_path = self.data_dir / "annotations" / "image_info.csv"
        
        # Split-specific directory mappings
        self.split_mappings = {
            'train': 'CarDD-TR',
            'val': 'CarDD-VAL', 
            'test': 'CarDD-TE'
        }
        
        # COCO annotation files
        self.coco_files = {
            'train': self.data_dir / "annotations" / "instances_train2017.json",
            'val': self.data_dir / "annotations" / "instances_val2017.json",
            'test': self.data_dir / "annotations" / "instances_test2017.json"
        }
        
        # Initialize data containers
        self.metadata_df = None
        self.coco_data = {}
        self.class_mapping = {}
        self.severity_mapping = {
            'minor': 0,
            'moderate': 1,
            'severe': 2
        }
        
        # Create output directories
        self.setup_output_structure()
        
        logger.info(f"Initialized COCO processor")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_output_structure(self):
        """Create YOLO dataset directory structure"""
        directories = [
            'train/images', 'train/labels', 'train/masks',
            'val/images', 'val/labels', 'val/masks',
            'test/images', 'test/labels', 'test/masks',
            'analysis', 'configs', 'visualizations'
        ]
        
        for dir_path in directories:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Created output directory structure")
    
    def get_split_directory(self, split, modality):
        """
        Get the correct directory path for a given split and modality
        
        Args:
            split (str): Dataset split ('train', 'val', 'test')
            modality (str): Data modality ('images', 'edges', 'masks')
            
        Returns:
            Path: Directory path for the specific split and modality
        """
        if split not in self.split_mappings:
            raise ValueError(f"Unknown split: {split}")
        
        split_folder = self.split_mappings[split]
        
        if modality == 'images':
            return self.data_dir / split_folder / f"{split_folder}-Image"
        elif modality == 'edges':
            return self.data_dir / split_folder / f"{split_folder}-Edge"
        elif modality == 'masks':
            return self.data_dir / split_folder / f"{split_folder}-Mask"
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def get_file_path_with_correct_extension(self, source_dir, image_name, modality):
        """
        Get the correct file path with the appropriate extension for each modality
        
        Args:
            source_dir (Path): Source directory
            image_name (str): Original image filename (e.g., '000001.jpg')
            modality (str): Data modality ('images', 'edges', 'masks')
            
        Returns:
            Path: File path with correct extension
        """
        base_name = Path(image_name).stem  # Get filename without extension
        
        if modality == 'images':
            # Images are in .jpg format
            return source_dir / f"{base_name}.jpg"
        elif modality in ['edges', 'masks']:
            # Edges and masks are in .png format
            return source_dir / f"{base_name}.png"
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def load_metadata(self):
        """Load and analyze image metadata"""
        logger.info("Loading image metadata...")
        
        try:
            self.metadata_df = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded {len(self.metadata_df)} metadata entries")
            
            # Analyze metadata distribution
            self.analyze_metadata(create_visualizations=False)
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
    
    def analyze_metadata(self, create_visualizations=True):
        """Analyze metadata statistics"""
        logger.info("Analyzing metadata distribution...")
        
        # Shooting angle distribution
        angle_dist = self.metadata_df['shooting angle'].value_counts()
        logger.info(f"Shooting angles: {dict(angle_dist)}")
        
        # Completeness distribution
        complete_dist = self.metadata_df['complete or partial'].value_counts()
        logger.info(f"Completeness: {dict(complete_dist)}")
        
        # Image dimensions
        avg_width = self.metadata_df['width'].mean()
        avg_height = self.metadata_df['height'].mean()
        logger.info(f"Average image size: {avg_width:.0f}x{avg_height:.0f}")
        
        # Categories per image
        avg_categories = self.metadata_df['#categories'].mean()
        logger.info(f"Average categories per image: {avg_categories:.2f}")
        
        # Create visualizations only if requested
        if create_visualizations:
            self.visualize_metadata()
    
    def visualize_metadata(self):
        """Create metadata visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Shooting angle distribution
        angle_counts = self.metadata_df['shooting angle'].value_counts()
        axes[0, 0].bar(angle_counts.index, angle_counts.values)
        axes[0, 0].set_title('Shooting Angle Distribution')
        axes[0, 0].set_xlabel('Shooting Angle')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Completeness distribution
        complete_counts = self.metadata_df['complete or partial'].value_counts()
        axes[0, 1].pie(complete_counts.values, labels=complete_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Complete vs Partial Images')
        
        # Categories per image
        axes[1, 0].hist(self.metadata_df['#categories'], bins=20, alpha=0.7)
        axes[1, 0].set_title('Categories per Image Distribution')
        axes[1, 0].set_xlabel('Number of Categories')
        axes[1, 0].set_ylabel('Frequency')
        
        # Image size distribution
        axes[1, 1].scatter(self.metadata_df['width'], self.metadata_df['height'], alpha=0.6)
        axes[1, 1].set_title('Image Dimensions')
        axes[1, 1].set_xlabel('Width')
        axes[1, 1].set_ylabel('Height')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'analysis' / 'metadata_analysis.png', dpi=300)
        plt.close()
        
        logger.info("Saved metadata visualizations")
    
    def load_coco_annotations(self):
        """Load COCO annotation files"""
        logger.info("Loading COCO annotations...")
        
        for split, file_path in self.coco_files.items():
            if file_path.exists():
                logger.info(f"Loading {split} annotations from {file_path}")
                self.coco_data[split] = COCO(str(file_path))
                
                # Get category information
                categories = self.coco_data[split].loadCats(self.coco_data[split].getCatIds())
                
                if not self.class_mapping:  # Initialize class mapping from first split
                    self.class_mapping = {cat['id']: cat['name'] for cat in categories}
                    logger.info(f"Class mapping: {self.class_mapping}")
            else:
                logger.warning(f"COCO file not found: {file_path}")
    
    def estimate_severity_from_mask(self, mask_area, bbox_area, image_area):
        """
        Estimate damage severity based on mask and bounding box areas
        
        Args:
            mask_area (float): Area of segmentation mask
            bbox_area (float): Area of bounding box
            image_area (float): Total image area
            
        Returns:
            str: Severity level ('minor', 'moderate', 'severe')
        """
        # Calculate relative areas
        mask_ratio = mask_area / image_area
        bbox_ratio = bbox_area / image_area
        mask_bbox_ratio = mask_area / bbox_area if bbox_area > 0 else 0
        
        # Severity estimation logic
        if mask_ratio > 0.15 or bbox_ratio > 0.2:  # Large damage
            return 'severe'
        elif mask_ratio > 0.05 or bbox_ratio > 0.08:  # Medium damage
            return 'moderate'
        else:  # Small damage
            return 'minor'
    
    def convert_coco_to_yolo(self, split):
        """
        Convert COCO annotations to YOLO format
        
        Args:
            split (str): Dataset split ('train', 'val', 'test')
        """
        logger.info(f"Converting {split} split to YOLO format...")
        
        if split not in self.coco_data:
            logger.warning(f"No COCO data for {split} split")
            return
        
        coco = self.coco_data[split]
        image_ids = coco.getImgIds()
        
        converted_count = 0
        severity_stats = {'minor': 0, 'moderate': 0, 'severe': 0}
        
        for img_id in tqdm(image_ids, desc=f"Processing {split}"):
            img_info = coco.loadImgs(img_id)[0]
            image_name = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            image_area = img_width * img_height
            
            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)
            
            if not annotations:
                continue
            
            # Process each modality (Image, Edge, Mask)
            for modality in ['images', 'edges', 'masks']:
                source_dir = self.get_split_directory(split, modality)
                target_dir = self.output_dir / split / modality
                
                # Get file path with correct extension
                source_path = self.get_file_path_with_correct_extension(source_dir, image_name, modality)
                target_path = target_dir / image_name
                
                if source_path.exists():
                    if modality == 'images':
                        # Copy original image
                        import shutil
                        shutil.copy2(source_path, target_path)
                    elif modality == 'edges':
                        # Process edge image
                        self.process_edge_image(source_path, target_path)
                    elif modality == 'masks':
                        # Process mask image
                        self.process_mask_image(source_path, target_path, annotations, coco)
            
            # Create YOLO label file
            label_path = self.output_dir / split / 'labels' / f"{Path(image_name).stem}.txt"
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    # Get bounding box
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # Get class ID (YOLO uses 0-based indexing)
                    class_id = ann['category_id'] - 1  # Convert to 0-based
                    
                    # Estimate severity
                    mask_area = ann.get('area', w * h)  # Use mask area if available
                    bbox_area = w * h
                    severity = self.estimate_severity_from_mask(mask_area, bbox_area, image_area)
                    severity_stats[severity] += 1
                    
                    # Write YOLO annotation
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                    
                    # Save segmentation if available
                    if 'segmentation' in ann:
                        self.save_segmentation_mask(ann, img_width, img_height, 
                                                  self.output_dir / split / 'masks' / f"{Path(image_name).stem}.png")
            
            converted_count += 1
        
        logger.info(f"Converted {converted_count} images for {split} split")
        logger.info(f"Severity distribution: {severity_stats}")
    
    def process_edge_image(self, source_path, target_path):
        """Process edge-detected image"""
        try:
            # Load and process edge image
            edge_img = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
            if edge_img is not None:
                # Apply additional edge processing if needed
                cv2.imwrite(str(target_path), edge_img)
        except Exception as e:
            logger.warning(f"Failed to process edge image {source_path}: {e}")
    
    def process_mask_image(self, source_path, target_path, annotations, coco):
        """Process mask image with overlaid annotations"""
        try:
            # Load original mask if exists
            if source_path.exists():
                import shutil
                shutil.copy2(source_path, target_path)
            else:
                # Generate mask from annotations
                img_info = coco.loadImgs(annotations[0]['image_id'])[0]
                mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                
                for ann in annotations:
                    if 'segmentation' in ann:
                        rle = coco.annToRLE(ann)
                        seg_mask = coco_mask.decode(rle)
                        mask = np.maximum(mask, seg_mask * 255)
                
                cv2.imwrite(str(target_path), mask)
        except Exception as e:
            logger.warning(f"Failed to process mask image {source_path}: {e}")
    
    def save_segmentation_mask(self, annotation, img_width, img_height, mask_path):
        """Save segmentation mask from COCO annotation"""
        try:
            # Create mask from segmentation
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            if 'segmentation' in annotation:
                segmentation = annotation['segmentation']
                if isinstance(segmentation, list):
                    # Polygon format
                    for seg in segmentation:
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 255)
                elif isinstance(segmentation, dict):
                    # RLE format
                    rle = segmentation
                    mask = coco_mask.decode(rle) * 255
            
            cv2.imwrite(str(mask_path), mask)
        except Exception as e:
            logger.warning(f"Failed to save segmentation mask: {e}")
    
    def create_yolo_config(self):
        """Create YOLO configuration files"""
        logger.info("Creating YOLO configuration files...")
        
        # Dataset configuration
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.values())
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Saved dataset config to {config_path}")
        
        # Training configuration
        training_config = {
            'model': 'yolo11n.pt',
            'data': str(config_path),
            'epochs': 100,
            'batch': 16,
            'imgsz': 1000,
            'device': 'auto',
            'project': 'runs/detect',
            'name': 'car_damage_yolo11n',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        training_config_path = self.output_dir / 'configs' / 'training_config.yaml'
        with open(training_config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        logger.info(f"Saved training config to {training_config_path}")
    
    def save_dataset_info(self):
        """Save comprehensive dataset information"""
        logger.info("Saving dataset information...")
        
        dataset_info = {
            'dataset_name': 'Car Damage Detection Dataset',
            'created_date': pd.Timestamp.now().isoformat(),
            'total_images': len(self.metadata_df) if self.metadata_df is not None else 0,
            'num_classes': len(self.class_mapping),
            'class_mapping': self.class_mapping,
            'severity_levels': list(self.severity_mapping.keys()),
            'data_splits': {
                split: len(self.coco_data[split].getImgIds()) if split in self.coco_data else 0
                for split in ['train', 'val', 'test']
            },
            'modalities': ['images', 'edges', 'masks'],
            'annotation_format': 'COCO -> YOLO',
            'image_formats': ['.jpg', '.jpeg', '.png'],
            'dataset_statistics': {}
        }
        
        # Add metadata statistics if available
        if self.metadata_df is not None:
            dataset_info['dataset_statistics'] = {
                'shooting_angles': self.metadata_df['shooting angle'].value_counts().to_dict(),
                'completeness': self.metadata_df['complete or partial'].value_counts().to_dict(),
                'avg_image_width': float(self.metadata_df['width'].mean()),
                'avg_image_height': float(self.metadata_df['height'].mean()),
                'avg_categories_per_image': float(self.metadata_df['#categories'].mean())
            }
        
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Saved dataset info to {info_path}")
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        logger.info("Generating analysis report...")
        
        report = f"""# Car Damage Dataset Analysis Report

## Dataset Overview
- **Total Images**: {len(self.metadata_df) if self.metadata_df is not None else 'N/A'}
- **Number of Classes**: {len(self.class_mapping)}
- **Annotation Format**: COCO -> YOLO
- **Modalities**: Images, Edges, Masks

## Class Distribution
"""
        
        for class_id, class_name in self.class_mapping.items():
            report += f"- **{class_name}** (ID: {class_id})\n"
        
        if self.metadata_df is not None:
            report += f"""
## Metadata Statistics
- **Average Image Size**: {self.metadata_df['width'].mean():.0f} x {self.metadata_df['height'].mean():.0f}
- **Average Categories per Image**: {self.metadata_df['#categories'].mean():.2f}

### Shooting Angle Distribution
"""
            for angle, count in self.metadata_df['shooting angle'].value_counts().items():
                report += f"- **{angle}**: {count} images\n"
            
            report += "\n### Completeness Distribution\n"
            for completeness, count in self.metadata_df['complete or partial'].value_counts().items():
                report += f"- **{completeness}**: {count} images\n"
        
        report += f"""
## Data Splits
"""
        for split, coco_obj in self.coco_data.items():
            num_images = len(coco_obj.getImgIds())
            num_annotations = len(coco_obj.getAnnIds())
            report += f"- **{split.title()}**: {num_images} images, {num_annotations} annotations\n"
        
        report_path = self.output_dir / 'analysis' / 'dataset_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved analysis report to {report_path}")
    
    def process_dataset(self):
        """Main processing pipeline"""
        logger.info("Starting COCO dataset processing...")
        
        try:
            # Load data
            self.load_metadata()
            self.load_coco_annotations()
            
            # Convert to YOLO format
            for split in ['train', 'val', 'test']:
                if split in self.coco_data:
                    self.convert_coco_to_yolo(split)
            
            # Create configuration files
            self.create_yolo_config()
            self.save_dataset_info()
            self.generate_analysis_report()
            
            logger.info("Dataset processing completed successfully!")
            logger.info(f"YOLO dataset saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            raise


def main():
    """Main execution function"""
    # Configuration
    data_dir = "data/CarDD_COCO"
    output_dir = "yolo_dataset"
    
    # Initialize processor
    processor = CarDamageCOCOProcessor(data_dir, output_dir)
    
    # Process dataset
    processor.process_dataset()


if __name__ == "__main__":
    main()
