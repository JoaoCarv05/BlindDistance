import os
import sys
import threading
import time

import cv2
import numpy as np


class StereoCamera:

    def __init__(self, calibration_file, left_id=0, right_id=1):
       
        if not os.path.exists(calibration_file):
            print(f"[ERROR] Calibration file not found: {calibration_file}")
            sys.exit(1)

        print(f"[INFO] Loading stereo calibration from: {calibration_file}")
        fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)

        self.image_w = int(fs.getNode("image_width").real())
        self.image_h = int(fs.getNode("image_height").real())

        # Rectification maps (pre-computed for fast cv2.remap)
        self.map1x = fs.getNode("map1x").mat()
        self.map1y = fs.getNode("map1y").mat()
        self.map2x = fs.getNode("map2x").mat()
        self.map2y = fs.getNode("map2y").mat()

        Q = fs.getNode("Q").mat()

        self.focal_length = Q[2, 3]
        if Q[3, 2] != 0:
            self.baseline = abs(1.0 / Q[3, 2])
        else:
            T = fs.getNode("T").mat()
            self.baseline = abs(T[0, 0])

        fs.release()
        print(f"[INFO] Calibration loaded: {self.image_w}x{self.image_h}, "
              f"focal={self.focal_length:.1f}px, baseline={self.baseline:.1f}mm")

        self.depth_h = self.image_h
        self.depth_w = self.image_w
        self.color_h = self.image_h
        self.color_w = self.image_w

        # ── Open cameras ─────────────────────────────────────────────────
        print(f"[INFO] Opening cameras (left={left_id}, right={right_id})...")
        self.cap_left = cv2.VideoCapture(left_id)
        self.cap_right = cv2.VideoCapture(right_id)

        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            print("[ERROR] Failed to open one or both cameras.")
            sys.exit(1)

        for cap in (self.cap_left, self.cap_right):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_h)

        print("[OK] Both cameras opened.")

        # ── Initialize StereoSGBM ────────────────────────────────────────
        block_size = 9
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128, 
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self._use_wls = False
        try:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
                matcher_left=self.stereo
            )
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.5)
            self._use_wls = True
            print("[INFO] WLS disparity filter enabled (opencv-contrib available).")
        except AttributeError:
            print("[INFO] WLS filter not available (install opencv-contrib-python "
                  "for smoother depth). Using raw StereoSGBM output.")

        # ── Thread-safe frame buffers ────────────────────────────────────
        self.latest_depth = None
        self.latest_color = None
        self._lock = threading.Lock()
        self.running = True

        # Start capture + processing thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[OK] Stereo capture thread started.")

    def _capture_loop(self):
        """Continuously capture, rectify, and compute depth."""
        while self.running:
            try:
                ret_l, frame_l = self.cap_left.read()
                ret_r, frame_r = self.cap_right.read()

                if not ret_l or not ret_r:
                    time.sleep(0.01)
                    continue

                # ── Rectify ──────────────────────────────────────────────
                rect_l = cv2.remap(
                    frame_l, self.map1x, self.map1y, cv2.INTER_LINEAR
                )
                rect_r = cv2.remap(
                    frame_r, self.map2x, self.map2y, cv2.INTER_LINEAR
                )

                # ── Convert to grayscale for disparity ───────────────────
                gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

                # ── Compute disparity ────────────────────────────────────
                if self._use_wls:
                    # Compute both left and right disparities for WLS filtering
                    disp_left = self.stereo.compute(gray_l, gray_r)
                    disp_right = self.right_matcher.compute(gray_r, gray_l)
                    disparity = self.wls_filter.filter(
                        disp_left, gray_l, None, disp_right
                    )
                else:
                    disparity = self.stereo.compute(gray_l, gray_r)

                disparity_float = disparity.astype(np.float32) / 16.0

                depth_mm = np.zeros_like(disparity_float, dtype=np.uint16)
                valid = disparity_float > 0
                depth_mm[valid] = (
                    (self.focal_length * self.baseline) / disparity_float[valid]
                ).astype(np.uint16)

                # Clamp unreasonably large values (> 10 meters)
                depth_mm[depth_mm > 10000] = 0

                # ── Store results ────────────────────────────────────────
                with self._lock:
                    self.latest_depth = depth_mm
                    self.latest_color = rect_l  # use rectified left image as color

            except Exception as e:
                print(f"[ERROR] Stereo capture thread: {e}")
                self.running = False
                break

    def get_frames(self):
        with self._lock:
            if self.latest_depth is None or self.latest_color is None:
                return None, None
            return self.latest_depth.copy(), self.latest_color.copy()

    def stop(self):
        """Release camera resources and stop the capture thread."""
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.cap_left.release()
        self.cap_right.release()
        print("[INFO] Stereo cameras released.")


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test StereoCamera output")
    parser.add_argument("--calibration", type=str,
                        default="stereo_calibration_data.xml",
                        help="Path to calibration file")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    args = parser.parse_args()

    cam = StereoCamera(args.calibration, args.left, args.right)
    print("\n[TEST] Showing depth + color. Press 'q' to quit.\n")

    try:
        while True:
            depth, color = cam.get_frames()
            if depth is None:
                time.sleep(0.01)
                continue

            # Visualize depth as a color map
            depth_vis = cv2.convertScaleAbs(depth, alpha=(255.0 / 4000.0))
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Show stats
            valid = depth[depth > 0]
            if valid.size > 0:
                min_d = np.min(valid)
                mean_d = np.mean(valid)
                cv2.putText(depth_vis, f"Min: {min_d}mm  Mean: {mean_d:.0f}mm",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

            combined = np.hstack([color, depth_vis])
            cv2.imshow("StereoCamera Test — Color | Depth", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
