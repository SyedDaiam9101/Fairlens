"""Detectify Evaluation Metrics - mAP, precision, recall."""
from typing import Any, Optional

import numpy as np


def compute_iou(box1: dict, box2: dict) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: Dict with x1, y1, x2, y2.
        box2: Dict with x1, y1, x2, y2.
        
    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_precision_recall(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Compute precision and recall for a set of predictions.
    
    Args:
        predictions: List of prediction dicts with bbox, class_id, confidence.
        ground_truths: List of ground truth dicts with bbox, class_id.
        iou_threshold: IoU threshold for matching.
        
    Returns:
        Tuple of (precision, recall).
    """
    if not predictions:
        return 0.0, 0.0
    if not ground_truths:
        return 0.0, 0.0
    
    # Sort predictions by confidence
    preds = sorted(predictions, key=lambda x: x.get("confidence", 0), reverse=True)
    
    matched_gt = set()
    tp = 0
    fp = 0
    
    for pred in preds:
        pred_class = pred.get("class_id")
        pred_bbox = pred.get("bbox", {})
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue
            if gt.get("class_id") != pred_class:
                continue
            
            gt_bbox = gt.get("bbox", {})
            iou = compute_iou(pred_bbox, gt_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truths) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def compute_ap(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    num_points: int = 11,
) -> float:
    """
    Compute Average Precision for a single class.
    
    Args:
        predictions: List of predictions for one class.
        ground_truths: List of ground truths for one class.
        iou_threshold: IoU threshold for matching.
        num_points: Number of recall points for interpolation.
        
    Returns:
        Average Precision value.
    """
    if not ground_truths:
        return 0.0
    
    # Sort by confidence
    preds = sorted(predictions, key=lambda x: x.get("confidence", 0), reverse=True)
    
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched_gt = set()
    
    for i, pred in enumerate(preds):
        pred_bbox = pred.get("bbox", {})
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue
            
            gt_bbox = gt.get("bbox", {})
            iou = compute_iou(pred_bbox, gt_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1
    
    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Append sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Compute AP using 11-point interpolation
    recall_levels = np.linspace(0, 1, num_points)
    ap = 0.0
    
    for r in recall_levels:
        prec_at_recall = precisions[recalls >= r]
        if len(prec_at_recall) > 0:
            ap += prec_at_recall.max()
    
    ap /= num_points
    return ap


def compute_map(
    all_predictions: dict[int, list[dict]],
    all_ground_truths: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute mean Average Precision across all classes.
    
    Args:
        all_predictions: Dict mapping class_id to list of predictions.
        all_ground_truths: Dict mapping class_id to list of ground truths.
        iou_threshold: IoU threshold for matching.
        
    Returns:
        Dict with mAP, per-class APs, and counts.
    """
    all_classes = set(all_predictions.keys()) | set(all_ground_truths.keys())
    
    aps = {}
    for class_id in all_classes:
        preds = all_predictions.get(class_id, [])
        gts = all_ground_truths.get(class_id, [])
        
        ap = compute_ap(preds, gts, iou_threshold)
        aps[class_id] = ap
    
    map_value = np.mean(list(aps.values())) if aps else 0.0
    
    return {
        "mAP": float(map_value),
        "per_class_ap": aps,
        "iou_threshold": iou_threshold,
        "num_classes": len(all_classes),
    }
