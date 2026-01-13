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
            - clip: numpy array of shape (frames_per_clip, num_points, 4) (float32)
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
        """Parse an ASCII .pcd file and return Nx4 numpy array of [x,y,z,gray].

        This parser reads the header until 'DATA ascii' and then parses subsequent
        lines. It expects at least x y z. If RGB is available it will compute a
        grayscale value (0-1) and use that as the 4th channel. RGB may be
        provided either as a packed integer (single column) or as three
        separate columns. If RGB is missing, the 4th channel will be 0.
        """
        pts = []
        with open(pcd_path, 'r') as f:
            header_ended = False
            for line in f:
                line = line.strip()
                if not header_ended:
                    if line.upper().startswith('DATA'):
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
                except ValueError:
                    continue

                # default grayscale
                gray = 0.0

                # try to detect RGB information
                # case A: three separate rgb columns present at indices 6,7,8 or 3,4,5 etc.
                use_rgb_triplet = False
                if len(parts) >= 9:
                    try:
                        r = float(parts[6])
                        g = float(parts[7])
                        b = float(parts[8])
                        use_rgb_triplet = True
                    except Exception:
                        use_rgb_triplet = False

                if use_rgb_triplet:
                    # normalize assuming 0-255 range
                    r = max(0.0, min(255.0, r))
                    g = max(0.0, min(255.0, g))
                    b = max(0.0, min(255.0, b))
                    gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                else:
                    # case B: packed integer in one column (commonly column 6)
                    if len(parts) >= 7:
                        try:
                            packed = float(parts[6])
                            packed_int = int(packed)
                            r = (packed_int >> 16) & 0xFF
                            g = (packed_int >> 8) & 0xFF
                            b = packed_int & 0xFF
                            gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                        except Exception:
                            # leave gray as default 0.0
                            pass

                pts.append((x, y, z, gray))

        if len(pts) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(pts, dtype=np.float32)

    def _sample_points(self, pts):
        """Sample exactly `self.num_points` from pts (np.array Nx4)."""
        n = pts.shape[0]
        if n == 0:
            # return zeros if empty
            return np.zeros((self.num_points, 4), dtype=np.float32)
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
        seq = self.sequences[idx]
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
            pts = self._parse_pcd_ascii(pcd_path)  # (P,4)
            sampled = self._sample_points(pts)  # (num_points,4)
            clip_frames.append(sampled)

        # clip shape: (frames_per_clip, num_points, 4)
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

        if len(sentences) == 0:
            sentence = ''
        else:
            if self.train:
                sentence = random.choice(sentences)
            else:
                sentence = sentences[0]

        return clip, sentence, idx


