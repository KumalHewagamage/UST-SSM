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

        # Store actual video metadata here
        self.video_data = [] 
        # Store (video_idx, start_frame_idx) tuples here
        self.samples = [] 
        
        self._load_data()

    def _load_data(self):
        search_path = os.path.join(self.root, '*')
        all_paths = glob.glob(search_path)
        
        # Hardcoded sampling interval (stride)
        step = 1 
        
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
                    
                    # Pre-load file paths to calculate windows
                    pcd_files = sorted(glob.glob(os.path.join(path, '*.pcd')))
                    n_frames = len(pcd_files)
                    
                    if n_frames > 0:
                        video_idx = len(self.video_data)
                        
                        # --- Sliding Window Logic ---
                        if self.split == 'train':
                            # TRAIN: Dense Sliding Window
                            if n_frames >= self.frames_per_clip:
                                for t in range(0, n_frames - self.frames_per_clip + 1, step):
                                    self.samples.append((video_idx, t))
                            else:
                                self.samples.append((video_idx, 0))
                        else:
                            # TEST: Single Centered Clip (Disabled sliding window)
                            if n_frames >= self.frames_per_clip:
                                # Calculate start index to center the clip
                                center_start = (n_frames - self.frames_per_clip) // 2
                                self.samples.append((video_idx, center_start))
                            else:
                                self.samples.append((video_idx, 0))

                        # Store metadata
                        self.video_data.append({
                            'path': path, 
                            'label': label,
                            'files': pcd_files
                        })
                        
            except (IndexError, ValueError):
                continue
                
        print(f"[{self.split.upper()}] Loaded {len(self.video_data)} videos, expanded to {len(self.samples)} clips.")

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
        # Retrieve the pre-calculated indices
        video_idx, start_frame = self.samples[idx]
        
        video_info = self.video_data[video_idx]
        files = video_info['files']
        label = video_info['label']
        
        n_frames = len(files)
        
        # Calculate indices based on sliding window start
        indices = [min(start_frame + i, n_frames - 1) for i in range(self.frames_per_clip)]
        
        clips = []
        for i in indices:
            pts = self._parse_pcd(files[i])
            clips.append(self._sample(pts))
            
        clip = np.stack(clips).astype(np.float32) 
        return clip, label