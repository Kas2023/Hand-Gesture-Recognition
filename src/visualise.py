import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dataloader import HandGestureDataset
from model import MultiTaskHandModel
from utils import load_checkpoint
import json
import seaborn as sns
import time

VIS_CONFIG = {
    "test_dir": "../dataset_test",
    "model_version": 4,
    "model_path": "../weights/best_model_v4.pth",
    "test_metrics_path": "../results/test_metrics_v4.json",
    "log_path": "../results/training_log_v4.json",
    "output_dir": "../results/visuals",
    "use_depth": True,
    "num_samples": 10,  # Number of sample images to generate
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gesture_names": ['call', 'dislike', 'like', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'three']
}

def denormalize_image(img_tensor):
    """Convert ImageNet normalized Tensor back to displayable RGB image"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * std + mean) * 255.0
    return img.astype(np.uint8)

@torch.no_grad()
def generate_visuals():
    """Randomly select test samples and generate classification, segmentation, and detection visualization results"""
    save_dir = os.path.join(VIS_CONFIG['output_dir'], f"v{VIS_CONFIG['model_version']}{'_no_depth' if not VIS_CONFIG['use_depth'] else ''}")
    os.makedirs(save_dir, exist_ok=True)

    model = MultiTaskHandModel(use_depth=VIS_CONFIG['use_depth'], version=VIS_CONFIG['model_version']).to(VIS_CONFIG['device'])
    load_checkpoint(model, None, None, VIS_CONFIG['model_path'], VIS_CONFIG['device'])
    model.eval()

    test_ds = HandGestureDataset(VIS_CONFIG['test_dir'], mode='val')
    
    # Randomly select a few samples
    indices = np.random.choice(len(test_ds), VIS_CONFIG['num_samples'], replace=False)

    for i, idx in enumerate(indices):
        batch = test_ds[idx]
        rgb_t = batch['rgb'].unsqueeze(0).to(VIS_CONFIG['device'])
        depth_t = batch['depth'].unsqueeze(0).to(VIS_CONFIG['device'])
        
        cls_p, det_p, seg_p = model(rgb_t, depth_t)
        
        pred_label = torch.argmax(cls_p, dim=1).item()
        gt_label = batch['label'].item()
        
        pred_mask = (seg_p[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        gt_mask = (batch['mask'][0].numpy() > 0.5).astype(np.uint8)
        
        pred_bbox = det_p[0].cpu().numpy() # [x1, y1, x2, y2] 0-1
        gt_bbox = batch['bbox'].numpy()

        # Convert image for display
        orig_img = denormalize_image(batch['rgb'])
        vis_img = orig_img.copy()
        h, w = vis_img.shape[:2]

        # --- Draw segmentation Mask (green semi-transparent overlay) ---
        mask_overlay = np.zeros_like(vis_img)
        mask_overlay[pred_mask > 0] = [0, 255, 0]
        vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.4, 0)

        # --- Draw BBox (red rectangle) ---
        x1, y1, x2, y2 = (pred_bbox * [w, h, w, h]).astype(int)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- Draw GT BBox (blue rectangle) ---
        # x1, y1, x2, y2 = (gt_bbox * [w, h, w, h]).astype(int)
        # cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # --- Add text labels ---
        text = f"Pred: {VIS_CONFIG['gesture_names'][pred_label]} | GT: {VIS_CONFIG['gesture_names'][gt_label]}"
        color = (0, 255, 0) if pred_label == gt_label else (0, 0, 255)
        cv2.putText(vis_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Save result
        save_path = os.path.join(save_dir, f"result_{i}.png")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth Mask")
        plt.imshow(orig_img)
        plt.imshow(gt_mask, alpha=0.3, cmap='jet')
        
        plt.subplot(1, 2, 2)
        plt.title("Prediction")
        plt.imshow(vis_img)
        
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

def plot_training_curves():
    """Plot training Loss and validation metrics curves"""
    if not os.path.exists(VIS_CONFIG['log_path']):
        print("Log file not found!")
        return

    with open(VIS_CONFIG['log_path'], 'r') as f:
        log = json.load(f)

    epochs = [item['epoch'] for item in log]
    
    # Extract Loss
    total_loss = [item['train_loss']['total'] for item in log]
    cls_loss = [item['train_loss']['cls'] for item in log]
    det_loss = [item['train_loss']['det'] for item in log]
    seg_loss = [item['train_loss']['seg'] for item in log]

    # Extract Metrics
    cls_acc = [item['val_metrics']['cls_top1_acc'] for item in log]
    seg_iou = [item['val_metrics']['seg_miou'] for item in log]
    det_acc = [item['val_metrics']['det_acc_at_05'] for item in log]

    # Find Phase switch point
    phase_switch = 0
    for i, item in enumerate(log):
        if item['phase'] == 2:
            phase_switch = i + 1
            break

    plt.figure(figsize=(15, 5))

    # Subplot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, total_loss, label='Total Loss', linewidth=2)
    plt.plot(epochs, cls_loss, '--', label='Cls Loss')
    plt.plot(epochs, det_loss, '--', label='Det Loss')
    plt.plot(epochs, seg_loss, '--', label='Seg Loss')
    if phase_switch > 0:
        plt.axvline(x=phase_switch, color='r', linestyle=':', label='Phase 2 Start')
    plt.title('Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Metrics curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, cls_acc, label='Cls Accuracy', color='green', linewidth=2)
    plt.plot(epochs, seg_iou, label='Seg mIoU', color='orange', linewidth=2)
    plt.plot(epochs, det_acc, label='Det Acc@0.5', color='blue', linewidth=2)
    if phase_switch > 0:
        plt.axvline(x=phase_switch, color='r', linestyle=':', label='Phase 2 Start')
    plt.title('Validation Metrics Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Score (0-1)')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(VIS_CONFIG['output_dir'], f"training_curves_v{VIS_CONFIG['model_version']}{'_no_depth' if not VIS_CONFIG['use_depth'] else ''}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")

def plot_confusion_matrix():
    """Plot test set classification confusion matrix"""
    if not os.path.exists(VIS_CONFIG['test_metrics_path']):
        print("Test metrics file not found!")
        return

    with open(VIS_CONFIG['test_metrics_path'], 'r') as f:
        metrics = json.load(f)

    cm = np.array(metrics['conf_matrix'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=VIS_CONFIG['gesture_names'],
                yticklabels=VIS_CONFIG['gesture_names'])
    
    plt.title('Normalized Confusion Matrix (Test Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    save_path = os.path.join(VIS_CONFIG['output_dir'], f"confusion_matrix_v{VIS_CONFIG['model_version']}{'_no_depth' if not VIS_CONFIG['use_depth'] else ''}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

@torch.no_grad()
def find_and_save_errors():
    """ Find and visualize all misclassified samples in the test set """
    save_dir = os.path.join(VIS_CONFIG['save_dir'], "misclassified_examples")
    os.makedirs(save_dir, exist_ok=True)
    
    model = MultiTaskHandModel(use_depth=VIS_CONFIG['use_depth'], version=VIS_CONFIG['model_version']).to(VIS_CONFIG['device'])
    load_checkpoint(model, None, None, VIS_CONFIG['model_path'], VIS_CONFIG['device'])
    model.eval()

    test_ds = HandGestureDataset(VIS_CONFIG['test_dir'], mode='val')
    
    print("Searching for misclassified examples...")
    error_count = 0
    
    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        start_time = time.time()
        rgb_t = sample['rgb'].unsqueeze(0).to(VIS_CONFIG['device'])
        depth_t = sample['depth'].unsqueeze(0).to(VIS_CONFIG['device'])
        gt_label = sample['label'].item()
        
        cls_p, _, _ = model(rgb_t, depth_t)
        pred_label = torch.argmax(cls_p, dim=1).item()
        end_time = time.time()
        print(f"Processed idx {idx} in {end_time - start_time:.2f}s - GT: {VIS_CONFIG['gesture_names'][gt_label]}, Pred: {VIS_CONFIG['gesture_names'][pred_label]}")
        exit(0)

        # If prediction is wrong, save image
        if pred_label != gt_label:
            error_count += 1
            orig_img = denormalize_image(sample['rgb'])
            
            vis_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            
            gt_name = VIS_CONFIG['gesture_names'][gt_label]
            pred_name = VIS_CONFIG['gesture_names'][pred_label]
            
            # Annotate error information on the image
            info_text = f"GT: {gt_name} | Pred: {pred_name}"
            cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            save_path = os.path.join(save_dir, f"error_{idx}_{gt_name}_as_{pred_name}.png")
            cv2.imwrite(save_path, vis_img)
            
            print(f"Found error at idx {idx}: GT '{gt_name}' misclassified as '{pred_name}'")
            
            if error_count >= VIS_CONFIG['num_samples']:
                break

    print(f"Done! Saved {error_count} error examples to {save_dir}")

if __name__ == "__main__":
    generate_visuals()
    plot_training_curves()
    plot_confusion_matrix()
    find_and_save_errors()