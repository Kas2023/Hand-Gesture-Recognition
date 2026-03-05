import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import random
from dataloader import HandGestureDataset
from model import MultiTaskHandModel
from utils import AverageMeter, calculate_metrics, save_checkpoint

CONFIG = {
    "seed": 42,
    "root_dir": "../dataset",
    "batch_size": 16,
    "warmup_epochs": 5,
    "epochs": 50,
    "lr": 1e-4,
    "val_weights": [0.4, 0.3, 0.3],
    "img_size": (224, 224),
    "model_version": 4,
    "use_depth": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "../weights/best_model_v4.pth",
    "results_path": "../results/training_log_v4.json"
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def complete_iou_loss(preds, targets):
    # Calculate basic IoU
    eps = 1e-7
    
    # Predicted box coordinates
    p_x1, p_y1, p_x2, p_y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    # Ground truth box coordinates
    t_x1, t_y1, t_x2, t_y2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    # Intersection area
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union area
    p_w = torch.clamp(p_x2 - p_x1, min=0)
    p_h = torch.clamp(p_y2 - p_y1, min=0)
    t_w = torch.clamp(t_x2 - t_x1, min=0)
    t_h = torch.clamp(t_y2 - t_y1, min=0)

    p_area = p_w * p_h
    t_area = t_w * t_h
    union_area = p_area + t_area - inter_area + eps
    
    iou = inter_area / union_area

    # Calculate center point distance rho^2(b, b_gt)
    p_cx, p_cy = (p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2
    t_cx, t_cy = (t_x1 + t_x2) / 2, (t_y1 + t_y2) / 2
    dist_sq = (p_cx - t_cx)**2 + (p_cy - t_cy)**2

    # Calculate the diagonal distance c^2 of the minimum bounding rectangle
    cw_x1 = torch.min(p_x1, t_x1)
    cw_y1 = torch.min(p_y1, t_y1)
    cw_x2 = torch.max(p_x2, t_x2)
    cw_y2 = torch.max(p_y2, t_y2)
    diag_sq = (cw_x2 - cw_x1)**2 + (cw_y2 - cw_y1)**2 + eps

    # Calculate aspect ratio consistency v and weight alpha
    # v = (4/pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2
    atan_diff = torch.atan(t_w / (t_h + eps)) - torch.atan(p_w / (p_h + eps))
    v = (4 / (torch.pi ** 2)) * atan_diff.pow(2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # Final CIoU formula
    ciou = iou - (dist_sq / diag_sq + alpha * v)
    
    return 1 - ciou.mean() # Return Loss (1 - CIoU)

def dice_loss(pred, target, smooth=1.):
    # pred: [N, 1, H, W], target: [N, 1, H, W]
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

class UncertaintyLoss(nn.Module):
    """Automatic weighted balance loss function (Core implementation of Innovation 20 points)"""
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_cls, loss_det, loss_seg):
        # Calculate weighted losses with uncertainty respectively
        # loss[i] / (2 * sigma^2) + log(sigma)
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * loss_cls + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * loss_det + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2 * loss_seg + self.log_vars[2]

        return loss0 + loss1 + loss2, [loss0.item(), loss1.item(), loss2.item()]

def train_epoch(model, loader, optimizer, criterion_mtl, device, phase=1):
    model.train()
    meters = {k: AverageMeter() for k in ['total', 'cls', 'det', 'seg']}
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device)
        bboxes = batch['bbox'].to(device)
        optimizer.zero_grad()
        cls_p, det_p, seg_p = model(rgb, depth)

        l_cls = nn.CrossEntropyLoss()(cls_p, labels)
        if CONFIG['model_version'] <= 3:
            l_det = nn.SmoothL1Loss()(det_p, bboxes)
        elif CONFIG['model_version'] <= 5:
            l_det = complete_iou_loss(det_p, bboxes)
        else:
            l_det = complete_iou_loss(det_p, bboxes) + 0.2 * nn.SmoothL1Loss()(det_p, bboxes)
        if CONFIG['model_version'] <= 5:
            l_seg = nn.BCELoss()(seg_p, masks)
        else:
            l_seg = nn.BCELoss()(seg_p, masks) + dice_loss(seg_p, masks)

        if phase == 1:
            # Phase 1: focus only on classification, direct backward
            total_loss = l_cls
            split_losses = [l_cls.item(), 0, 0]
        else:
            # Phase 2: joint training, using learnable weight balancing
            total_loss, split_losses = criterion_mtl(l_cls, l_det, l_seg)
        
        total_loss.backward()
        optimizer.step()

        # Record
        meters['total'].update(total_loss.item())
        meters['cls'].update(split_losses[0])
        meters['det'].update(split_losses[1])
        meters['seg'].update(split_losses[2])
        pbar.set_postfix(loss=meters['total'].avg)

    return {k: v.avg for k, v in meters.items()}

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    # Store all predictions and ground truth for calculating final metrics (IoU, F1, Accuracy)
    all_results = []
    
    for batch in tqdm(loader, desc="Validating"):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        
        cls_p, det_p, seg_p = model(rgb, depth)
        
        all_results.append({
            'cls_p': cls_p.cpu(),
            'det_p': det_p.cpu(),
            'seg_p': seg_p.cpu(),
            'cls_gt': batch['label'],
            'det_gt': batch['bbox'],
            'seg_gt': batch['mask']
        })

    metrics = calculate_metrics(all_results)
    return metrics

def main():
    # 0. Fix random seed
    seed_everything(CONFIG['seed'])

    # 1. Prepare directories
    os.makedirs("../weights", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    # 2. Load data
    full_dataset = HandGestureDataset(CONFIG['root_dir'], mode='train', img_size=CONFIG['img_size'])
    # 80/20 split train and validation sets
    train_size = int(0.8 * len(full_dataset))
    all_samples = full_dataset.samples
    random.shuffle(all_samples)
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:]
    train_ds = HandGestureDataset(CONFIG['root_dir'], samples=train_samples, mode='train', img_size=CONFIG['img_size'])
    val_ds = HandGestureDataset(CONFIG['root_dir'], samples=val_samples, mode='val', img_size=CONFIG['img_size'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 3. Model and initialization
    model = MultiTaskHandModel(use_depth=CONFIG['use_depth'], version=CONFIG['model_version']).to(CONFIG['device'])
    criterion_mtl = UncertaintyLoss().to(CONFIG['device'])
    
    best_val = 0.0
    history = []

    # ==========================================================================
    # Phase 1: Warm up backbone and classification (Backbone + Cls Head)
    # ==========================================================================
    print("\n>>> Starting Phase 1: Warming up Backbone & Classification...")

    # 1. Gradient control
    for name, param in model.named_parameters():
        if 'det_head' in name or 'seg_head' in name:
            param.requires_grad = False  # Freeze spatial localization parameters
        else:
            param.requires_grad = True   # Unfreeze Backbone and classification head
    
    for param in criterion_mtl.parameters():
        param.requires_grad = False     # Phase 1: Do not update automatic weight parameters

    # 2. Rebuild optimizer (only includes parameters that need to be updated in Phase 1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=CONFIG['lr']
    )

    # 3. Run Phase 1 loop
    for epoch in range(CONFIG['warmup_epochs']):
        print(f"\nPhase 1 - Epoch {epoch+1}/{CONFIG['warmup_epochs']}")
        
        # Note: Added a phase parameter to inform train_epoch to only calculate CE
        train_loss = train_epoch(model, train_loader, optimizer, criterion_mtl, CONFIG['device'], phase=1)
        val_metrics = validate(model, val_loader, CONFIG['device'])
        
        print(f"Train Loss: {train_loss['total']:.4f} | Val Cls Acc: {val_metrics['cls_top1_acc']:.4f} | Val Seg IoU: {val_metrics['seg_miou']:.4f} | Val Det ACC: {val_metrics['det_acc_at_05']:.4f}")
        
        # Phase 1 uses classification accuracy as the criterion
        if val_metrics['cls_top1_acc'] > best_val:
            best_val = val_metrics['cls_top1_acc']
            save_checkpoint(model, optimizer, criterion_mtl, epoch, CONFIG['save_path'])
            print(f"Best model saved (Val Acc: {best_val:.4f})")

        history.append({"epoch": epoch + 1, "phase": 1, "train_loss": train_loss, "val_metrics": val_metrics})
        with open(CONFIG['results_path'], 'w') as f:
            json.dump(history, f, indent=4)

    # ==========================================================================
    # Phase 2: Multi-task joint optimization
    # ==========================================================================
    print("\n>>> Starting Phase 2: Full Multi-task Training...")
    
    for param in model.parameters():
        param.requires_grad = True
    for param in criterion_mtl.parameters():
        param.requires_grad = True
    
    best_val = 0.0

    optimizer = optim.AdamW([
        # Previously trained parts: use lower learning rate (protection)
        {'params': model.rgb_backbone.parameters(), 'lr': CONFIG['lr'] * 0.5},
        {'params': model.cls_head.parameters(), 'lr': CONFIG['lr'] * 0.5},
        
        # Newly trained parts: use standard/higher learning rate (catch up)
        {'params': model.det_head.parameters(), 'lr': CONFIG['lr']},
        # {'params': model.seg_head.parameters(), 'lr': CONFIG['lr']},
        {'params': model.seg_up1.parameters(), 'lr': CONFIG['lr']},
        {'params': model.seg_up2.parameters(), 'lr': CONFIG['lr']},
        {'params': model.seg_up3.parameters(), 'lr': CONFIG['lr']},
        {'params': model.seg_final.parameters(), 'lr': CONFIG['lr']},
        
        # Automatic weight parameters: use highest learning rate (rapid balancing)
        {'params': criterion_mtl.parameters(), 'lr': CONFIG['lr'] * 10}
    ], lr=CONFIG['lr'])

    for epoch in range(CONFIG['warmup_epochs'], CONFIG['epochs']):
        print(f"\nPhase 2 - Epoch {epoch+1}/{CONFIG['epochs']}")
        
        # At this time phase=2, calculate all losses and use criterion_mtl for automatic weighting
        train_loss = train_epoch(model, train_loader, optimizer, criterion_mtl, CONFIG['device'], phase=2)
        val_metrics = validate(model, val_loader, CONFIG['device'])
        
        print(f"Train Loss: {train_loss['total']:.4f} | Val Cls Acc: {val_metrics['cls_top1_acc']:.4f} | Val Seg IoU: {val_metrics['seg_miou']:.4f} | Val Det ACC: {val_metrics['det_acc_at_05']:.4f}")

        # Phase 2 saves the model based on comprehensive score
        # Calculate comprehensive score
        cls_score = (val_metrics['cls_top1_acc'] + val_metrics['cls_f1_macro']) / 2
        seg_score = (val_metrics['seg_miou'] + val_metrics['seg_dice']) / 2
        det_score = (val_metrics['det_mean_iou'] + val_metrics['det_acc_at_05']) / 2
        
        # Comprehensive weighted score
        combined_score = CONFIG['val_weights'][0] * cls_score + CONFIG['val_weights'][1] * seg_score + CONFIG['val_weights'][2] * det_score

        print(f"Phase 2 Score: {combined_score:.4f} (Cls: {cls_score:.2f}, Seg: {seg_score:.2f}, Det: {det_score:.2f})")

        # Save the model based on comprehensive score
        if combined_score > best_val:
            best_val = combined_score
            save_checkpoint(model, optimizer, criterion_mtl, epoch, CONFIG['save_path'])
            print(f"Best balanced model saved! Score: {best_val:.4f}")

        history.append({"epoch": epoch + 1, "phase": 2, "train_loss": train_loss, "val_metrics": val_metrics})
        with open(CONFIG['results_path'], 'w') as f:
            json.dump(history, f, indent=4)

if __name__ == "__main__":
    main()