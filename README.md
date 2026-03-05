# Hand Gesture Recognition Multi-Task Learning Model

A multi-task learning model for hand gesture recognition with classification, segmentation, and detection tasks. This project uses ConvNeXt-Tiny as the backbone and supports both RGB and depth information.

## Table of Contents

- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [Model Versions](#model-versions)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)

---


## Setup & Installation

### Installation Steps

1. **Clone or download the project**

```bash
cd /path/to/project
```

2. **Create a conda environment (optional but recommended)**

```bash
conda create -n hand_gesture python=3.10
conda activate hand_gesture
```

3. **Install dependencies**

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

4. **Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Dataset Preparation

Organize your data in the following structure:

```
dataset/
  student_name_1/
    G01_call/
      clip01/
        rgb/
        depth_raw/
        annotation/
        depth_metadata.json
      clip02/
      ...
    G02_dislike/
    ...
  student_name_2/
  ...

dataset_test/
  (same structure as above)
```

- `rgb/`: RGB images (.png)
- `depth_raw/`: Depth maps (.npy files)
- `annotation/`: Hand segmentation masks (.png)
- 10 gesture classes: call, dislike, like, ok, one, palm, peace, rock, stop, three

---

## Project Structure

```
src/
├── train.py           # Training script with 2-phase training strategy
├── evaluate.py        # Model evaluation on test set
├── visualise.py       # Visualization and error analysis
├── model.py           # Multi-task model architecture
├── dataloader.py      # Data loading and augmentation
└── utils.py           # Utility functions (metrics, checkpoint management)

weights/               # Trained model checkpoints
results/               # Training logs and metrics
  ├── training_log_*.json
  ├── test_metrics_*.json
  └── visuals/         # Visualization outputs
```

---

## Model Versions

The codebase supports 6 different model versions. The table below lists the changes from each version compared to the previous one:

| Version | Classification | Detection | Segmentation | Use Case |
|---------|----------------|-----------|--------------|----------|
| 1 | Simple head, CE | Simple head, SmoothL1 | Simple decoder, BCE | Deprecated |
| 2 | Enhanced head | -- | -- | Baseline |
| 3 | -- | -- | Skip connection decoder | Better segmentation |
| 4 | -- | Enhanced head, CIoU | -- | Better detection, **Recommended** |
| 5 | -- | SmoothL1 | -- | Less stable |
| 6 | -- | Combined loss (+CIoU) | Combined loss (+Dice) | Similar performance to v4 |

---

## Training

### Basic Usage

```bash
cd src
python train.py
```

The training will:
1. Train the selected version of the model
2. Save logs to `results_path` in `CONFIG`, default `../results/training_log_v4.json`
3. Save best model checkpoint to `save_path` in `CONFIG`, default `../weights/best_model_v4.pth`
4. Print metrics every epoch:
   - Classification accuracy
   - Segmentation mIoU
   - Detection accuracy@0.5 IoU

### Modifiable Parameters

Edit the `CONFIG` dictionary in `train.py` to customize training:

```python
CONFIG = {
    # Data configuration
    "root_dir": "../dataset",              # Path to training dataset
    "img_size": (224, 224),                # Input image size (no need for revision)
    
    # Training parameters
    "batch_size": 16,                      # Batch size (adjust based on GPU memory)
    "warmup_epochs": 5,                    # Phase 1: classification warmup epochs
    "epochs": 50,                          # Total training epochs
    "lr": 1e-4,                            # Initial learning rate
    
    # Model configuration
    "model_version": 4,                    # Model version (1-6, 4 is recommended)
    "use_depth": True,                     # Whether to  use depth information
    
    # Model saving in Phase 2
    "val_weights": [0.4, 0.3, 0.3],        # To calculate comprehensive score
    
    # Random seed
    "seed": 42,                            # Random seed for reproducibility
    
    # Device and paths
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "../weights/best_model_v4.pth",
    "results_path": "../results/training_log_v4.json"
}
```

---

## Evaluation

### Basic Usage

```bash
cd src
python evaluate.py
```

This will:
1. Load the selected trained model
2. Run inference on test set
3. Calculate and save metrics to `results_path` in `TEST_CONFIG`, default `../results/test_metrics_v4.json`
4. Print test set results

### Modifiable Parameters

Edit the `TEST_CONFIG` dictionary in `evaluate.py`:

```python
TEST_CONFIG = {
    # Data
    "test_dir": "../dataset_test", # Path to test dataset
    
    # Model (must match training)
    "model_version": 4,
    "use_depth": True,
    "model_path": "../weights/best_model_v4.pth",
    
    # Inference
    "batch_size": 16,
    
    # Device and output
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "results_path": "../results/test_metrics_v4.json"
}
```

### Evaluation Metrics

The evaluation outputs the following metrics in JSON format:

```json
{
    "cls_top1_acc": 0.95,              // Classification top-1 accuracy
    "cls_f1_macro": 0.94,              // Macro-averaged F1 score
    "seg_miou": 0.82,                  // Segmentation mean IoU
    "seg_dice": 0.88,                  // Segmentation Dice coefficient
    "det_mean_iou": 0.75,              // Detection mean bounding box IoU
    "det_acc_at_05": 0.71,             // Detection accuracy@0.5 IoU
    "conf_matrix": [[...], [...], ...] // 10x10 confusion matrix
}
```

### Performance Inference

For average inference time per image:

```python
# In evaluate.py, call after evaluate_test_set():
get_average_inference_time()  # Prints FPS and inference time
```

---

## Visualization

### Generate Visualizations

```bash
cd src
python visualise.py
```

This will automatically:
1. Generate random sample visualizations
2. Plot training curves (loss and metrics)
3. Plot confusion matrix
4. Search for and save misclassified examples

### Modifiable Parameters

Edit the `VIS_CONFIG` dictionary in `visualise.py`:

```python
VIS_CONFIG = {
    # Data
    "test_dir": "../dataset_test",
    
    # Model (must match training)
    "model_version": 4,
    "use_depth": True,
    "model_path": "../weights/best_model_v4.pth",
    
    # Logs and results
    "test_metrics_path": "../results/test_metrics_v4.json",
    "log_path": "../results/training_log_v4.json",
    "output_dir": "../results/visuals",
    
    # Visualization parameters
    "num_samples": 10,     # Number of samples to visualize
    
    # Gesture names (10 classes in order)
    "gesture_names": ['call', 'dislike', 'like', 'ok', 'one', 
                      'palm', 'peace', 'rock', 'stop', 'three'],
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
```

### Output Files

The visualization generates:

- **`v4/result_0.png`, `v4/result_1.png`, etc.**
  - Side-by-side comparison of ground truth and predictions
  - Green overlay for predicted segmentation mask
  - Red bounding box for detected region
  - Text label showing predicted vs ground truth gesture

- **`training_curves_v4.png`**
  - Left: Total loss, classification loss, detection loss, segmentation loss
  - Right: Classification accuracy, segmentation mIoU, detection accuracy
  - Vertical line marks Phase 2 start

- **`confusion_matrix_v4.png`**
  - Normalized confusion matrix for classification task
  - Shows which gestures are confused with each other

- **`misclassified_examples/`**
  - Examples of incorrectly classified gestures
  - Filenames show ground truth and predicted labels
