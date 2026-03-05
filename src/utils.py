import torch
import os
from sklearn.metrics import f1_score, confusion_matrix

class AverageMeter:
    """Track average values during training"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def calculate_bbox_iou(box1, box2):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)
    return iou

def calculate_metrics(results):
    cls_p = torch.cat([r['cls_p'] for r in results], dim=0)   # [N, 10]
    det_p = torch.cat([r['det_p'] for r in results], dim=0)   # [N, 4]
    seg_p = torch.cat([r['seg_p'] for r in results], dim=0)   # [N, 1, H, W]
    
    cls_gt = torch.cat([r['cls_gt'] for r in results], dim=0)
    det_gt = torch.cat([r['det_gt'] for r in results], dim=0)
    seg_gt = torch.cat([r['seg_gt'] for r in results], dim=0)

    # --- Classification Metrics ---
    # Top-1 Accuracy
    preds_cls = torch.argmax(cls_p, dim=1)
    acc = (preds_cls == cls_gt).float().mean().item()
    
    # Macro-averaged F1 & Confusion Matrix
    y_true = cls_gt.cpu().numpy()
    y_pred = preds_cls.cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='macro')
    conf_mat = confusion_matrix(y_true, y_pred)

    # --- Segmentation Metrics ---
    seg_p_bin = (seg_p > 0.5).float()
    seg_gt_bin = (seg_gt > 0.5).float()
    
    intersection = (seg_p_bin * seg_gt_bin).sum(dim=(1, 2, 3))
    union = (seg_p_bin + seg_gt_bin).sum(dim=(1, 2, 3)) - intersection
    
    # Mean IoU
    mIoU = (intersection / (union + 1e-6)).mean().item()
    
    # Dice Coefficient
    dice = (2. * intersection / (union + intersection + 1e-6)).mean().item()

    # --- Detection Metrics ---
    # Mean BBox IoU
    bbox_ious = calculate_bbox_iou(det_p, det_gt)
    mean_bbox_iou = bbox_ious.mean().item()
    
    # Detection Accuracy @ 0.5 IoU
    det_acc_05 = (bbox_ious > 0.5).float().mean().item()


    return {
        "cls_top1_acc": acc,
        "cls_f1_macro": f1,
        "seg_miou": mIoU,
        "seg_dice": dice,
        "det_mean_iou": mean_bbox_iou,
        "det_acc_at_05": det_acc_05,
        "conf_matrix": conf_mat.tolist() # Convert to list for easy JSON saving
    }

def save_checkpoint(model, optimizer, criterion_mtl, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mtl_state_dict': criterion_mtl.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, criterion_mtl, path, device):
    if not os.path.exists(path):
        print("No checkpoint found.")
        return 0
        
    checkpoint = torch.load(path, map_location=device)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if criterion_mtl is not None:
        criterion_mtl.load_state_dict(checkpoint['mtl_state_dict'])
    
    return checkpoint['epoch']