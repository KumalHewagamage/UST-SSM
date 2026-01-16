import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class MSRAction3D(Dataset):
    """
    MSR Action3D Dataset for Point Cloud Action Recognition.
    Returns: (clip, label)
    """
    def __init__(self, root, frames_per_clip=24, num_points=2048, split='train'):
        super().__init__()
        self.root = root
        self.frames_per_clip = frames_per_clip
        self.num_points = num_points
        self.split = split.lower()
        
        if self.split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")

        self.samples = [] 
        self._load_data()

    def _load_data(self):
        search_path = os.path.join(self.root, '*')
        all_paths = glob.glob(search_path)
        
        count = 0
        for path in all_paths:
            if not os.path.isdir(path): continue

            filename = os.path.basename(os.path.normpath(path))
            try:
                # Filename format: aXX_sXX_eXX
                parts = filename.split('_')
                label = int(parts[0][1:]) - 1    # a01 -> 0
                subject_id = int(parts[1][1:])   # s01 -> 1
                
                # Split Criteria: Train (1-5), Test (6+)
                is_train_subject = (subject_id <= 5)
                
                if (self.split == 'train' and is_train_subject) or \
                   (self.split == 'test' and not is_train_subject):
                    
                    pcd_files = glob.glob(os.path.join(path, '*.pcd'))
                    if len(pcd_files) > 0:
                        self.samples.append({'path': path, 'label': label})
                        count += 1
            except (IndexError, ValueError):
                continue
        print(f"[{self.split.upper()}] Loaded {count} sequences.")

    def _parse_pcd(self, pcd_path):
        xyz = []
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('DATA'):
                    start_idx = i + 1
                    break
            
            for line in lines[start_idx:]:
                vals = line.strip().split()
                if len(vals) >= 3:
                    xyz.append([float(vals[0]), float(vals[1]), float(vals[2])])
            
            return np.array(xyz, dtype=np.float32)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)

    def _sample(self, pts):
        n = pts.shape[0]
        if n == 0: return np.zeros((self.num_points, 3), dtype=np.float32)
        
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            choice = np.random.choice(n, self.num_points, replace=True)
            return pts[choice, :]
        return pts[idx, :]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        files = sorted(glob.glob(os.path.join(sample['path'], '*.pcd')))
        
        # Temporal Sampling
        n_frames = len(files)
        mid = n_frames // 2
        start = max(0, mid - (self.frames_per_clip // 2))
        indices = [min(start + i, n_frames - 1) for i in range(self.frames_per_clip)]
        
        clips = []
        for i in indices:
            pts = self._parse_pcd(files[i])
            clips.append(self._sample(pts))
            
        clip = np.stack(clips).astype(np.float32) 
        return clip, sample['label']