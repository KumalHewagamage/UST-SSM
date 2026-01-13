import os
import sys
import json
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class HumanML3D(Dataset):
    """Dataset for HumanML3D-style sequences stored as per-sequence folders.

    Directory structure expected for `root`:
      sequence_000001/
        frame_000.pcd
        frame_001.pcd
        ...
        report.json

    For each sequence this dataset returns:
      - clip: numpy array of shape (frames_per_clip, num_points, 3) (float32)
      - sentence: string (random sentence from report description during train, first sentence during test)
      - index: integer sequence index

    Frame sampling: choose `frames_per_clip` frames centered around the middle frame of the sequence.
    Point sampling: per-frame random sampling to `num_points` (repeat if necessary).
    """

    def __init__(self, root, frames_per_clip=16, num_points=2048, train=True):
        super(HumanML3D, self).__init__()
        self.root = root
        self.frames_per_clip = frames_per_clip
        self.num_points = num_points
        self.train = train
        # decide which folder to use: if root contains 'train' and 'test' subfolders,
        # use the appropriate one according to the `train` flag. Otherwise use root.
        seq_root = root
        train_dir = os.path.join(root, 'train')
        test_dir = os.path.join(root, 'test')
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            seq_root = train_dir if self.train else test_dir

        # collect sequences (each sequence is a folder)
        seq_dirs = [os.path.join(seq_root, d) for d in os.listdir(seq_root)
                    if os.path.isdir(os.path.join(seq_root, d))]
        seq_dirs = sorted(seq_dirs)

        self.sequences = []  # list of dicts: {frames: [paths], report: path}
        for sd in seq_dirs:
            # collect pcd frames
            frames = sorted(glob.glob(os.path.join(sd, 'frame_*.pcd')))
            report_path = os.path.join(sd, 'report.json')
            if len(frames) == 0:
                continue
            if not os.path.exists(report_path):
                # still include sequences without report.json but set report to None
                report_path = None
            self.sequences.append({'dir': sd, 'frames': frames, 'report': report_path})

    def __len__(self):
        return len(self.sequences)

    def _parse_pcd_ascii(self, pcd_path):
        """Parse an ASCII .pcd file and return Nx3 numpy array of xyz.

        This is a lightweight parser that reads the header until 'DATA ascii' and
        then parses subsequent lines assuming first three columns are x y z.
        """
        xyz_list = []
        with open(pcd_path, 'r') as f:
            header_ended = False
            for line in f:
                line = line.strip()
                if not header_ended:
                    if line.startswith('DATA'):
                        # expecting 'DATA ascii'
                        header_ended = True
                    continue
                if line == '':
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    xyz_list.append((x, y, z))
                except ValueError:
                    # skip malformed lines
                    continue
        if len(xyz_list) == 0:
            print(f"Warning: no points found in {pcd_path}", file=sys.stderr)
            pass
        return np.array(xyz_list, dtype=np.float32)

    def _sample_points(self, pts):
        """Sample exactly `self.num_points` from pts (np.array Nx3)."""
        n = pts.shape[0]
        if n >= self.num_points:
            idx = np.random.choice(n, size=self.num_points, replace=False)
            return pts[idx, :]
        else:
            repeat, residue = divmod(self.num_points, n)
            if residue > 0:
                r = np.random.choice(n, size=residue, replace=False)
                idxs = np.concatenate([np.arange(n) for _ in range(repeat)] + [r], axis=0)
            else:
                idxs = np.concatenate([np.arange(n) for _ in range(repeat)], axis=0)
            return pts[idxs, :]

    def __getitem__(self, idx):
        # Try to load sample, skip corrupted ones
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                current_idx = (idx + attempt) % len(self.sequences)
                seq = self.sequences[current_idx]
                frames = seq['frames']
                num_frames_available = len(frames)

                # center anchor frame
                mid = num_frames_available // 2
                start = mid - (self.frames_per_clip // 2)
                # build indices clamped to [0, num_frames_available-1]
                indices = [min(max(0, start + i), num_frames_available - 1) for i in range(self.frames_per_clip)]

                clip_frames = []
                for fi in indices:
                    pcd_path = frames[fi]
                    pts = self._parse_pcd_ascii(pcd_path)  # (P,3)
                    
                    # Skip if no points found
                    if pts.shape[0] == 0:
                        print(f"Warning: no points found in {pcd_path}")
                        raise ValueError(f"Empty point cloud: {pcd_path}")
                    
                    sampled = self._sample_points(pts)  # (num_points,3)
                    clip_frames.append(sampled)

                # clip shape: (frames_per_clip, num_points, 3)
                clip = np.stack(clip_frames, axis=0).astype(np.float32)

                # read report.json and extract sentences
                sentences = []
                if seq['report'] is not None:
                    try:
                        with open(seq['report'], 'r') as f:
                            j = json.load(f)
                            desc = j.get('description', '')
                            # split by '.' and strip
                            parts = [s.strip() for s in desc.split('.')]
                            sentences = [s for s in parts if len(s) > 0]
                    except Exception:
                        sentences = []

                # Ensure we have at least one sentence
                if len(sentences) == 0:
                    sentences = ["No description available"]
                
                if self.train:
                    sentence = random.choice(sentences)
                else:
                    sentence = sentences[0]

                return clip, sentence
                
            except (ValueError, ZeroDivisionError, IndexError, FileNotFoundError) as e:
                # Skip corrupted sample and try next one
                if attempt == 0:
                    print(f"Warning: Skipping corrupted sample at index {current_idx}: {e}")
                continue
        
        # If all attempts fail, return a dummy sample
        print(f"Error: Could not load valid sample after {max_attempts} attempts, returning dummy data")
        dummy_clip = np.zeros((self.frames_per_clip, self.num_points, 3), dtype=np.float32)
        dummy_sentence = "dummy sample"
        return dummy_clip, dummy_sentence


