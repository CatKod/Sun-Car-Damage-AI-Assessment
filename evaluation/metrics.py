"""
Comprehensive Evaluation Metrics for Car Damage Analysis
======================================================

This module provides comprehensive evaluation metrics for detection,
segmentation, and severity estimation models.

Features:
- Standard detection metrics (mAP, precision, recall)
- Segmentation metrics (IoU, Dice score)
- Severity estimation metrics
- Robustness analysis by metadata attributes
- Visualization and reporting utilities

Author: AI Engineer
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import cv2
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, confusion_matrix,
    classification_report
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionMetrics:
    """Metrics for object detection evaluation."""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.true_positives = []
        self.false_positives = []
        self.false_negatives = []
        self.confidences = []
        self.ious = []
        
    def add_detections(self, pred_boxes: List, pred_scores: List, pred_classes: List,
                      gt_boxes: List, gt_classes: List):
        """
        Add detection results for evaluation.
        
        Args:
            pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
            pred_scores: List of confidence scores
            pred_classes: List of predicted class IDs
            gt_boxes: List of ground truth bounding boxes [x1, y1, x2, y2]
            gt_classes: List of ground truth class IDs
        """
        if not pred_boxes and not gt_boxes:
            return
            
        if not gt_boxes:  # No ground truth, all predictions are false positives
            self.false_positives.extend([1] * len(pred_boxes))
            self.true_positives.extend([0] * len(pred_boxes))
            self.confidences.extend(pred_scores)
            return
            
        if not pred_boxes:  # No predictions, all ground truth are false negatives
            self.false_negatives.extend([1] * len(gt_boxes))
            return
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(pred_boxes, gt_boxes)
        
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_boxes)
        
        # Sort predictions by confidence (descending)
        pred_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        
        for pred_idx in pred_indices:
            pred_class = pred_classes[pred_idx]
            confidence = pred_scores[pred_idx]
            
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(gt_boxes)):
                if gt_matched[gt_idx] or gt_classes[gt_idx] != pred_class:
                    continue
                    
                iou = iou_matrix[pred_idx][gt_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Determine if it's a true positive
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                self.true_positives.append(1)
                self.false_positives.append(0)
                gt_matched[best_gt_idx] = True
                self.ious.append(best_iou)
            else:
                self.true_positives.append(0)
                self.false_positives.append(1)
                self.ious.append(best_iou)
            
            self.confidences.append(confidence)
        
        # Count false negatives (unmatched ground truth)
        fn_count = sum(1 for matched in gt_matched if not matched)
        self.false_negatives.extend([1] * fn_count)
    
    def _calculate_iou_matrix(self, boxes1: List, boxes2: List) -> List[List[float]]:
        """Calculate IoU matrix between two sets of boxes."""
        iou_matrix = []
        
        for box1 in boxes1:
            iou_row = []
            for box2 in boxes2:
                iou = self._calculate_iou(box1, box2)
                iou_row.append(iou)
            iou_matrix.append(iou_row)
        
        return iou_matrix
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ap(self, interpolated: bool = True) -> float:
        """Calculate Average Precision."""
        if not self.true_positives:
            return 0.0
        
        # Sort by confidence
        indices = sorted(range(len(self.confidences)), 
                        key=lambda i: self.confidences[i], reverse=True)
        
        tp = [self.true_positives[i] for i in indices]
        fp = [self.false_positives[i] for i in indices]
        
        # Calculate cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        total_positives = tp_cumsum[-1] + sum(self.false_negatives)
        
        if total_positives == 0:
            return 0.0
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_positives
        
        if interpolated:
            # 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                precision_interp = np.max(precision[recall >= t]) if np.any(recall >= t) else 0.0
                ap += precision_interp / 11
        else:
            # Exact calculation
            ap = 0.0
            for i in range(1, len(recall)):
                ap += (recall[i] - recall[i-1]) * precision[i]
        
        return ap
    
    def get_metrics(self) -> Dict[str, float]:
        """Get comprehensive detection metrics."""
        if not self.true_positives:
            return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mean_iou': 0.0}
        
        tp_total = sum(self.true_positives)
        fp_total = sum(self.false_positives)
        fn_total = sum(self.false_negatives)
        
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        ap = self.calculate_ap()
        mean_iou = np.mean(self.ious) if self.ious else 0.0
        
        return {
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': mean_iou,
            'total_tp': tp_total,
            'total_fp': fp_total,
            'total_fn': fn_total
        }

class SegmentationMetrics:
    """Metrics for instance segmentation evaluation."""
    
    def __init__(self):
        self.ious = []
        self.dice_scores = []
        self.pixel_accuracies = []
    
    def add_masks(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """
        Add mask pair for evaluation.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
        """
        # Ensure binary masks
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        # Calculate IoU
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        
        # Calculate Dice score
        dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
        
        # Calculate pixel accuracy
        correct_pixels = (pred_mask == gt_mask).sum()
        total_pixels = pred_mask.size
        pixel_accuracy = correct_pixels / total_pixels
        
        self.ious.append(iou)
        self.dice_scores.append(dice)
        self.pixel_accuracies.append(pixel_accuracy)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get segmentation metrics."""
        return {
            'mean_iou': np.mean(self.ious) if self.ious else 0.0,
            'mean_dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'mean_pixel_accuracy': np.mean(self.pixel_accuracies) if self.pixel_accuracies else 0.0,
            'std_iou': np.std(self.ious) if self.ious else 0.0,
            'std_dice': np.std(self.dice_scores) if self.dice_scores else 0.0
        }

class SeverityMetrics:
    """Metrics for severity estimation evaluation."""
    
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
        self.is_classification = None
    
    def add_predictions(self, predictions: List[float], ground_truth: List[float], 
                       is_classification: bool = True):
        """
        Add predictions for evaluation.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            is_classification: Whether this is classification (vs regression)
        """
        self.predictions.extend(predictions)
        self.ground_truth.extend(ground_truth)
        self.is_classification = is_classification
    
    def get_metrics(self) -> Dict[str, float]:
        """Get severity estimation metrics."""
        if not self.predictions:
            return {}
        
        if self.is_classification:
            return {
                'accuracy': accuracy_score(self.ground_truth, self.predictions),
                'precision_macro': precision_score(self.ground_truth, self.predictions, average='macro', zero_division=0),
                'recall_macro': recall_score(self.ground_truth, self.predictions, average='macro', zero_division=0),
                'f1_macro': f1_score(self.ground_truth, self.predictions, average='macro', zero_division=0),
                'precision_weighted': precision_score(self.ground_truth, self.predictions, average='weighted', zero_division=0),
                'recall_weighted': recall_score(self.ground_truth, self.predictions, average='weighted', zero_division=0),
                'f1_weighted': f1_score(self.ground_truth, self.predictions, average='weighted', zero_division=0)
            }
        else:
            return {
                'mae': mean_absolute_error(self.ground_truth, self.predictions),
                'rmse': np.sqrt(mean_squared_error(self.ground_truth, self.predictions)),
                'mse': mean_squared_error(self.ground_truth, self.predictions),
                'r2': 1 - (np.sum((np.array(self.ground_truth) - np.array(self.predictions))**2) / 
                          np.sum((np.array(self.ground_truth) - np.mean(self.ground_truth))**2))
            }

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for car damage analysis."""
    
    def __init__(self, output_dir: str = "evaluation/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics containers
        self.detection_metrics = defaultdict(lambda: DetectionMetrics())
        self.segmentation_metrics = defaultdict(lambda: SegmentationMetrics())
        self.severity_metrics = defaultdict(lambda: SeverityMetrics())
        
        # Metadata for robustness analysis
        self.metadata = {}
        self.results_by_metadata = defaultdict(list)
    
    def load_metadata(self, metadata_path: str):
        """Load image metadata for robustness analysis."""
        if Path(metadata_path).suffix == '.xlsx':
            self.metadata = pd.read_excel(metadata_path).set_index('image_name').to_dict('index')
        elif Path(metadata_path).suffix == '.csv':
            self.metadata = pd.read_csv(metadata_path).set_index('image_name').to_dict('index')
        else:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {len(self.metadata)} images")
    
    def add_detection_results(self, image_name: str, pred_boxes: List, pred_scores: List, 
                            pred_classes: List, gt_boxes: List, gt_classes: List, 
                            category_name: str = 'overall'):
        """Add detection evaluation results."""
        self.detection_metrics[category_name].add_detections(
            pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes
        )
        
        # Store for metadata analysis
        self.results_by_metadata[image_name] = {
            'detection': {
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores,
                'pred_classes': pred_classes,
                'gt_boxes': gt_boxes,
                'gt_classes': gt_classes
            }
        }
    
    def add_segmentation_results(self, image_name: str, pred_masks: List[np.ndarray], 
                               gt_masks: List[np.ndarray], category_name: str = 'overall'):
        """Add segmentation evaluation results."""
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            self.segmentation_metrics[category_name].add_masks(pred_mask, gt_mask)
        
        # Update metadata results
        if image_name not in self.results_by_metadata:
            self.results_by_metadata[image_name] = {}
        
        self.results_by_metadata[image_name]['segmentation'] = {
            'pred_masks': pred_masks,
            'gt_masks': gt_masks
        }
    
    def add_severity_results(self, image_name: str, predictions: List[float], 
                           ground_truth: List[float], is_classification: bool = True,
                           category_name: str = 'overall'):
        """Add severity estimation results."""
        self.severity_metrics[category_name].add_predictions(
            predictions, ground_truth, is_classification
        )
        
        # Update metadata results
        if image_name not in self.results_by_metadata:
            self.results_by_metadata[image_name] = {}
        
        self.results_by_metadata[image_name]['severity'] = {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'is_classification': is_classification
        }
    
    def evaluate_robustness(self) -> Dict[str, Dict]:
        """Evaluate model robustness by metadata attributes."""
        robustness_results = {}
        
        if not self.metadata:
            logger.warning("No metadata available for robustness analysis")
            return robustness_results
        
        # Group results by metadata attributes
        metadata_groups = defaultdict(list)
        
        for image_name, results in self.results_by_metadata.items():
            if image_name in self.metadata:
                meta = self.metadata[image_name]
                
                # Group by shooting angle
                if 'shooting_angle' in meta:
                    angle = meta['shooting_angle']
                    metadata_groups[f'angle_{angle}'].append((image_name, results))
                
                # Group by completeness
                if 'completeness' in meta:
                    completeness = meta['completeness']
                    metadata_groups[f'completeness_{completeness}'].append((image_name, results))
                
                # Group by damage type if available
                if 'damage_type' in meta:
                    damage_type = meta['damage_type']
                    metadata_groups[f'damage_type_{damage_type}'].append((image_name, results))
        
        # Evaluate each group
        for group_name, group_data in metadata_groups.items():
            if len(group_data) < 5:  # Skip small groups
                continue
            
            # Create temporary metrics for this group
            group_det_metrics = DetectionMetrics()
            group_seg_metrics = SegmentationMetrics()
            group_sev_metrics = SeverityMetrics()
            
            for image_name, results in group_data:
                # Detection metrics
                if 'detection' in results:
                    det = results['detection']
                    group_det_metrics.add_detections(
                        det['pred_boxes'], det['pred_scores'], det['pred_classes'],
                        det['gt_boxes'], det['gt_classes']
                    )
                
                # Segmentation metrics
                if 'segmentation' in results:
                    seg = results['segmentation']
                    for pred_mask, gt_mask in zip(seg['pred_masks'], seg['gt_masks']):
                        group_seg_metrics.add_masks(pred_mask, gt_mask)
                
                # Severity metrics
                if 'severity' in results:
                    sev = results['severity']
                    group_sev_metrics.add_predictions(
                        sev['predictions'], sev['ground_truth'], sev['is_classification']
                    )
            
            # Store group results
            robustness_results[group_name] = {
                'sample_count': len(group_data),
                'detection': group_det_metrics.get_metrics(),
                'segmentation': group_seg_metrics.get_metrics(),
                'severity': group_sev_metrics.get_metrics()
            }
        
        return robustness_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'overall_metrics': {},
            'category_metrics': {},
            'robustness_analysis': {},
            'failure_analysis': {}
        }
        
        # Overall metrics
        for metric_type, metrics_dict in [
            ('detection', self.detection_metrics),
            ('segmentation', self.segmentation_metrics),
            ('severity', self.severity_metrics)
        ]:
            if 'overall' in metrics_dict:
                report['overall_metrics'][metric_type] = metrics_dict['overall'].get_metrics()
        
        # Category-wise metrics
        for category in self.detection_metrics.keys():
            if category != 'overall':
                report['category_metrics'][category] = {
                    'detection': self.detection_metrics[category].get_metrics(),
                    'segmentation': self.segmentation_metrics[category].get_metrics(),
                    'severity': self.severity_metrics[category].get_metrics()
                }
        
        # Robustness analysis
        report['robustness_analysis'] = self.evaluate_robustness()
        
        # Failure analysis
        report['failure_analysis'] = self._analyze_failure_cases()
        
        # Save report
        with open(self.output_dir / 'comprehensive_evaluation.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_visualizations(report)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        logger.info(f"Comprehensive evaluation report saved to {self.output_dir}")
        return report
    
    def _analyze_failure_cases(self) -> Dict[str, Any]:
        """Analyze failure cases for insights."""
        failure_analysis = {
            'low_confidence_detections': [],
            'high_iou_misses': [],
            'severe_segmentation_errors': [],
            'severity_estimation_errors': []
        }
        
        # Analyze detection failures
        overall_det = self.detection_metrics.get('overall')
        if overall_det and overall_det.confidences:
            # Find low confidence true positives and high confidence false positives
            for i, (tp, conf) in enumerate(zip(overall_det.true_positives, overall_det.confidences)):
                if tp == 1 and conf < 0.3:  # Low confidence TP
                    failure_analysis['low_confidence_detections'].append({
                        'index': i,
                        'confidence': conf,
                        'type': 'low_confidence_tp'
                    })
                elif tp == 0 and conf > 0.7:  # High confidence FP
                    failure_analysis['low_confidence_detections'].append({
                        'index': i,
                        'confidence': conf,
                        'type': 'high_confidence_fp'
                    })
        
        return failure_analysis
    
    def _generate_visualizations(self, report: Dict[str, Any]):
        """Generate evaluation visualizations."""
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Overall metrics comparison
        overall = report['overall_metrics']
        if overall:
            metrics_names = []
            metrics_values = []
            
            for task, metrics in overall.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_names.append(f"{task}_{metric_name}")
                        metrics_values.append(value)
            
            if metrics_names:
                axes[0, 0].bar(range(len(metrics_names)), metrics_values)
                axes[0, 0].set_xticks(range(len(metrics_names)))
                axes[0, 0].set_xticklabels(metrics_names, rotation=45, ha='right')
                axes[0, 0].set_title('Overall Metrics')
                axes[0, 0].set_ylabel('Score')
        
        # 2. Robustness by shooting angle
        robustness = report['robustness_analysis']
        angle_groups = {k: v for k, v in robustness.items() if 'angle_' in k}
        
        if angle_groups:
            angles = [k.replace('angle_', '') for k in angle_groups.keys()]
            detection_aps = [v['detection'].get('ap', 0) for v in angle_groups.values()]
            
            axes[0, 1].bar(angles, detection_aps)
            axes[0, 1].set_title('Detection AP by Shooting Angle')
            axes[0, 1].set_xlabel('Shooting Angle')
            axes[0, 1].set_ylabel('Average Precision')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Robustness by completeness
        completeness_groups = {k: v for k, v in robustness.items() if 'completeness_' in k}
        
        if completeness_groups:
            completeness_levels = [k.replace('completeness_', '') for k in completeness_groups.keys()]
            segmentation_ious = [v['segmentation'].get('mean_iou', 0) for v in completeness_groups.values()]
            
            axes[0, 2].bar(completeness_levels, segmentation_ious)
            axes[0, 2].set_title('Segmentation IoU by Completeness')
            axes[0, 2].set_xlabel('Completeness Level')
            axes[0, 2].set_ylabel('Mean IoU')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Sample count by metadata groups
        if robustness:
            group_names = list(robustness.keys())
            sample_counts = [v['sample_count'] for v in robustness.values()]
            
            axes[1, 0].bar(group_names, sample_counts)
            axes[1, 0].set_title('Sample Count by Metadata Groups')
            axes[1, 0].set_xlabel('Group')
            axes[1, 0].set_ylabel('Sample Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Category-wise performance
        categories = report['category_metrics']
        if categories:
            cat_names = list(categories.keys())
            cat_detection_f1 = [v['detection'].get('f1', 0) for v in categories.values()]
            
            axes[1, 1].bar(cat_names, cat_detection_f1)
            axes[1, 1].set_title('Detection F1 by Category')
            axes[1, 1].set_xlabel('Category')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Failure analysis summary
        failures = report['failure_analysis']
        failure_types = ['Low Conf TP', 'High Conf FP', 'IoU Misses', 'Seg Errors']
        failure_counts = [
            len(failures.get('low_confidence_detections', [])),
            len([f for f in failures.get('low_confidence_detections', []) if f.get('type') == 'high_confidence_fp']),
            len(failures.get('high_iou_misses', [])),
            len(failures.get('severe_segmentation_errors', []))
        ]
        
        axes[1, 2].bar(failure_types, failure_counts)
        axes[1, 2].set_title('Failure Case Analysis')
        axes[1, 2].set_xlabel('Failure Type')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_visualizations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate markdown evaluation report."""
        md_content = f"""# Car Damage Analysis - Comprehensive Evaluation Report

## Executive Summary

This report provides a comprehensive evaluation of the car damage analysis system, including detection, segmentation, and severity estimation performance.

**Generated**: {report['timestamp']}

## Overall Performance

### Detection Metrics
"""
        
        if 'detection' in report['overall_metrics']:
            det_metrics = report['overall_metrics']['detection']
            md_content += f"""
- **Average Precision (AP)**: {det_metrics.get('ap', 0):.3f}
- **Precision**: {det_metrics.get('precision', 0):.3f}
- **Recall**: {det_metrics.get('recall', 0):.3f}
- **F1 Score**: {det_metrics.get('f1', 0):.3f}
- **Mean IoU**: {det_metrics.get('mean_iou', 0):.3f}
"""
        
        if 'segmentation' in report['overall_metrics']:
            seg_metrics = report['overall_metrics']['segmentation']
            md_content += f"""
### Segmentation Metrics
- **Mean IoU**: {seg_metrics.get('mean_iou', 0):.3f}
- **Mean Dice Score**: {seg_metrics.get('mean_dice', 0):.3f}
- **Pixel Accuracy**: {seg_metrics.get('mean_pixel_accuracy', 0):.3f}
"""
        
        if 'severity' in report['overall_metrics']:
            sev_metrics = report['overall_metrics']['severity']
            md_content += f"""
### Severity Estimation Metrics
- **Accuracy**: {sev_metrics.get('accuracy', 0):.3f}
- **F1 Score (Macro)**: {sev_metrics.get('f1_macro', 0):.3f}
- **MAE**: {sev_metrics.get('mae', 0):.3f}
"""
        
        # Robustness Analysis
        md_content += "\\n## Robustness Analysis\\n"
        
        robustness = report['robustness_analysis']
        if robustness:
            md_content += "\\n### Performance by Shooting Angle\\n"
            md_content += "| Angle | Samples | Detection AP | Segmentation IoU |\\n"
            md_content += "|-------|---------|--------------|------------------|\\n"
            
            for group_name, metrics in robustness.items():
                if 'angle_' in group_name:
                    angle = group_name.replace('angle_', '')
                    det_ap = metrics['detection'].get('ap', 0)
                    seg_iou = metrics['segmentation'].get('mean_iou', 0)
                    samples = metrics['sample_count']
                    md_content += f"| {angle} | {samples} | {det_ap:.3f} | {seg_iou:.3f} |\\n"
            
            md_content += "\\n### Performance by Completeness\\n"
            md_content += "| Completeness | Samples | Detection AP | Segmentation IoU |\\n"
            md_content += "|--------------|---------|--------------|------------------|\\n"
            
            for group_name, metrics in robustness.items():
                if 'completeness_' in group_name:
                    completeness = group_name.replace('completeness_', '')
                    det_ap = metrics['detection'].get('ap', 0)
                    seg_iou = metrics['segmentation'].get('mean_iou', 0)
                    samples = metrics['sample_count']
                    md_content += f"| {completeness} | {samples} | {det_ap:.3f} | {seg_iou:.3f} |\\n"
        
        # Category Performance
        if report['category_metrics']:
            md_content += "\\n## Performance by Category\\n"
            md_content += "| Category | Detection F1 | Segmentation IoU | Severity Acc |\\n"
            md_content += "|----------|--------------|------------------|--------------|\\n"
            
            for category, metrics in report['category_metrics'].items():
                det_f1 = metrics['detection'].get('f1', 0)
                seg_iou = metrics['segmentation'].get('mean_iou', 0)
                sev_acc = metrics['severity'].get('accuracy', 0)
                md_content += f"| {category} | {det_f1:.3f} | {seg_iou:.3f} | {sev_acc:.3f} |\\n"
        
        # Failure Analysis
        failures = report['failure_analysis']
        md_content += f"""
## Failure Analysis

- **Low Confidence True Positives**: {len([f for f in failures.get('low_confidence_detections', []) if f.get('type') == 'low_confidence_tp'])}
- **High Confidence False Positives**: {len([f for f in failures.get('low_confidence_detections', []) if f.get('type') == 'high_confidence_fp'])}
- **Severe Segmentation Errors**: {len(failures.get('severe_segmentation_errors', []))}

## Recommendations

Based on the evaluation results:

1. **Detection Improvements**: Focus on reducing false positives with high confidence scores
2. **Segmentation Refinement**: Improve mask boundary accuracy for better IoU scores
3. **Severity Estimation**: Enhance feature extraction for better severity prediction
4. **Robustness**: Address performance variations across different shooting angles and completeness levels

## Model Files and Outputs

- Evaluation visualizations: `evaluation_visualizations.png`
- Detailed metrics: `comprehensive_evaluation.json`
- Model predictions and analysis available in respective model directories

---
*Generated by Car Damage Analysis Evaluation Framework*
"""
        
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write(md_content)

def main():
    """Example usage of the evaluation framework."""
    evaluator = ComprehensiveEvaluator()
    
    # Load metadata (example)
    # evaluator.load_metadata('data/metadata.json')
    
    # Add some example results (in real usage, these come from model predictions)
    # evaluator.add_detection_results(...)
    # evaluator.add_segmentation_results(...)
    # evaluator.add_severity_results(...)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report()
    
    print("Evaluation completed. Check the 'evaluation/reports' directory for results.")

if __name__ == "__main__":
    main()
