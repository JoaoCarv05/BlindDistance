import os
import time
import cv2
import numpy as np


class DataRecorder:
    """
    Saves paired depth maps, color images, and grid distance arrays
    into label-organized directories for AI training data collection.

    Directory structure:
        dataset/
        ├── <label>/
        │   ├── depth/   <-- .npy files with uint16 480x640 depth in mm
        │   ├── color/   <-- .png files with BGR color image
        │   └── grid/    <-- .npy files with uint16 1D grid distance array

    Controls (handled externally in main loop):
        Keys 1-5  → select label
        r         → toggle recording on/off
    """

    LABELS = ['clear', 'person', 'wall', 'furniture', 'other_obstacle']
    LABEL_KEYS = {ord('1'): 0, ord('2'): 1, ord('3'): 2, ord('4'): 3, ord('5'): 4}

    def __init__(self, base_dir: str = 'dataset', save_fps: float = 2.0):
        self.base_dir = base_dir
        self.save_interval = 1.0 / save_fps
        self._last_save_time = 0.0
        self._counters: dict[str, int] = {}

        # Build directory tree and initialise counters from existing files
        for label in self.LABELS:
            for sub in ('depth', 'color', 'grid'):
                path = os.path.join(base_dir, label, sub)
                os.makedirs(path, exist_ok=True)

            # Count existing files to avoid overwriting previous sessions
            existing = os.listdir(os.path.join(base_dir, label, 'depth'))
            npy_files = [f for f in existing if f.endswith('.npy')]
            if npy_files:
                max_idx = max(int(os.path.splitext(f)[0]) for f in npy_files)
            else:
                max_idx = 0
            self._counters[label] = max_idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_frame(
        self,
        depth_img: np.ndarray,
        color_img: np.ndarray,
        grid_distances: list,
        label: str,
    ) -> bool:
        """
        Save a single frame if the rate-limit interval has elapsed.

        Parameters
        ----------
        depth_img       : 480×640 uint16 numpy array (depth in mm)
        color_img       : 480×640×3 uint8 numpy array (BGR)
        grid_distances  : list of uint16 distances sampled at grid points
        label           : one of DataRecorder.LABELS

        Returns True if the frame was actually written, False if skipped.
        """
        now = time.monotonic()
        if now - self._last_save_time < self.save_interval:
            return False
        if label not in self.LABELS:
            raise ValueError(f"Unknown label '{label}'. Valid: {self.LABELS}")

        self._last_save_time = now
        self._counters[label] += 1
        idx = self._counters[label]
        filename = f"{idx:08d}"

        # Depth map
        depth_path = os.path.join(self.base_dir, label, 'depth', f"{filename}.npy")
        np.save(depth_path, depth_img)

        # Color image
        color_path = os.path.join(self.base_dir, label, 'color', f"{filename}.png")
        cv2.imwrite(color_path, color_img)

        # Grid distances
        grid_path = os.path.join(self.base_dir, label, 'grid', f"{filename}.npy")
        np.save(grid_path, np.array(grid_distances, dtype=np.uint16))

        return True

    def get_frame_count(self, label: str) -> int:
        """Return total saved frames for the given label."""
        return self._counters.get(label, 0)

    def total_frames(self) -> int:
        """Return total saved frames across all labels."""
        return sum(self._counters.values())
