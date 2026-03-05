import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision.transforms.functional as TF

class HandGestureDataset(Dataset):
    def __init__(self, root_dir, samples=None, mode='train', img_size=(224, 224)):
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._load_samples()
        
        # Normalization parameters
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        samples = []
        student_dirs = [d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)]
        
        for s_dir in student_dirs:
            # Manually remove certain bad data
            if s_dir.endswith('_x'):
                continue
            for g_idx in range(1, 11):
                gesture_dir = glob.glob(os.path.join(s_dir, f'G{g_idx:02d}*'))
                if not gesture_dir: continue
                
                clips = glob.glob(os.path.join(gesture_dir[0], 'clip*'))
                for c_path in clips:
                    ann_dir = os.path.join(c_path, 'annotation')
                    if not os.path.exists(ann_dir): continue
                    
                    # Use annotation file name as reference to match RGB and Depth
                    ann_files = glob.glob(os.path.join(ann_dir, '*.png'))
                    for f_ann in ann_files:
                        fname = os.path.basename(f_ann)
                        f_rgb = os.path.join(c_path, 'rgb', fname)
                        f_depth = os.path.join(c_path, 'depth_raw', fname.replace('.png', '.npy'))
                        
                        if os.path.exists(f_rgb) and os.path.exists(f_depth):
                            samples.append({
                                'rgb': f_rgb,
                                'depth': f_depth,
                                'mask': f_ann,
                                'label': g_idx - 1 # 0-9
                            })
                        else:
                            print(f"Warning: Missing files for {fname} in {c_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(s['rgb']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        depth = np.load(s['depth']).astype(np.float32)
        
        # --- Data Augmentation ---
        if self.mode == 'train':
            # 1. Random rotation
            angle = random.uniform(-20, 20)
            img = TF.to_pil_image(img)
            img = TF.rotate(img, angle)
            mask = TF.to_pil_image(mask)
            mask = TF.rotate(mask, angle)
            # Depth simultaneous rotation
            depth_pil = TF.to_pil_image(depth)
            depth_pil = TF.rotate(depth_pil, angle)
            depth = np.array(depth_pil)
            
            # 2. Color jitter (RGB only)
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            img = color_jitter(img)
            
            img = np.array(img)
            mask = np.array(mask)

        # Resize
        img = self.rgb_transform(img)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_LINEAR)

        # Normalize
        mask = (mask > 0).astype(np.float32) 

        coords = np.argwhere(mask > 0)
        h, w = mask.shape
        if coords.size > 0:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            bbox = np.array([xmin/w, ymin/h, xmax/w, ymax/h], dtype=np.float32)
        else:
            bbox = np.array([0, 0, 0, 0], dtype=np.float32)
        
        depth = np.clip(depth, 0, 4000) / 4000.0
        depth = torch.from_numpy(depth).unsqueeze(0).to(torch.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).to(torch.float32)
        
        return {
            'rgb': img,
            'depth': depth,
            'mask': mask,
            'label': torch.tensor(s['label'], dtype=torch.long),
            'bbox': torch.tensor(bbox, dtype=torch.float32)
        }


if __name__ == "__main__":
    dataset = HandGestureDataset("../dataset_test")
    print(f"Total samples: {len(dataset)}")
    sample = dataset[0]
    print(f"RGB shape: {sample['rgb'].shape}, Depth shape: {sample['depth'].shape}, Mask shape: {sample['mask'].shape}")