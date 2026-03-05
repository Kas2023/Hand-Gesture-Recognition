import torch
from torch.utils.data import DataLoader
import json
import tqdm
import time
from dataloader import HandGestureDataset
from model import MultiTaskHandModel
from utils import calculate_metrics, load_checkpoint

TEST_CONFIG = {
    "test_dir": "../dataset_test",
    "batch_size": 16,
    "model_version": 4,
    "use_depth": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "../weights/best_model_v4.pth",
    "results_path": "../results/test_metrics_v4.json"
}

@torch.no_grad()
def evaluate_test_set():
    test_ds = HandGestureDataset(TEST_CONFIG['test_dir'], mode='val')
    test_loader = DataLoader(test_ds, batch_size=TEST_CONFIG['batch_size'], shuffle=False, num_workers=4)
    print(f"Loaded {len(test_ds)} test samples.")

    model = MultiTaskHandModel(version=TEST_CONFIG['model_version'], use_depth=TEST_CONFIG['use_depth']).to(TEST_CONFIG['device'])
    _ = load_checkpoint(model, None, None, TEST_CONFIG['model_path'], TEST_CONFIG['device'])
    model.eval()

    all_results = []
    print("Running inference on test set...")
    for batch in tqdm.tqdm(test_loader, desc="Evaluating"):
        rgb = batch['rgb'].to(TEST_CONFIG['device'])
        depth = batch['depth'].to(device=TEST_CONFIG['device'])
        
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
    
    print("\n--- TEST SET RESULTS ---")
    for k, v in metrics.items():
        if k != 'conf_matrix':
            print(f"{k}: {v:.4f}")

    with open(TEST_CONFIG['results_path'], 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nFull metrics saved to {TEST_CONFIG['results_path']}")

@torch.no_grad()
def get_average_inference_time():
    model = MultiTaskHandModel(use_depth=TEST_CONFIG['use_depth'], version=TEST_CONFIG['model_version']).to(TEST_CONFIG['device'])
    load_checkpoint(model, None, None, TEST_CONFIG['model_path'], TEST_CONFIG['device'])
    model.eval()

    # Prepare GPU memory data (excluding disk reading effects)
    dummy_rgb = torch.randn(1, 3, 224, 224).to(TEST_CONFIG['device'])
    dummy_depth = torch.randn(1, 1, 224, 224).to(TEST_CONFIG['device'])

    # GPU Warming up
    print("GPU Warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_rgb, dummy_depth)
    
    # Formal timing
    num_samples = 200
    torch.cuda.synchronize() # Ensure warmup instructions complete
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(dummy_rgb, dummy_depth)
    
    torch.cuda.synchronize() # Force CPU to wait for GPU to complete all calculations
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_samples
    print(f"Average Inference Time: {avg_time:.4f} seconds")
    print(f"FPS: {1/avg_time:.1f}")

if __name__ == "__main__":
    evaluate_test_set()
    get_average_inference_time()