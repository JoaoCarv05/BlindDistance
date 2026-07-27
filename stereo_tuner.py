
import argparse
import os
import sys
import time

import cv2
import numpy as np


def nothing(x):
    """Trackbar callback (required by OpenCV, does nothing)."""
    pass


def main():
    parser = argparse.ArgumentParser(description="StereoSGBM Parameter Tuner")
    parser.add_argument("--calibration", type=str,
                        default="stereo_calibration_data.xml",
                        help="Path to calibration file")
    parser.add_argument("--left", type=int, default=0,
                        help="Left camera device ID")
    parser.add_argument("--right", type=int, default=1,
                        help="Right camera device ID")
    args = parser.parse_args()

    # ── Load calibration ─────────────────────────────────────────────────
    if not os.path.exists(args.calibration):
        print(f"[ERROR] Calibration file not found: {args.calibration}")
        print("        Run stereo_calibration.py first.")
        sys.exit(1)

    fs = cv2.FileStorage(args.calibration, cv2.FILE_STORAGE_READ)
    image_w = int(fs.getNode("image_width").real())
    image_h = int(fs.getNode("image_height").real())
    map1x = fs.getNode("map1x").mat()
    map1y = fs.getNode("map1y").mat()
    map2x = fs.getNode("map2x").mat()
    map2y = fs.getNode("map2y").mat()
    Q = fs.getNode("Q").mat()
    fs.release()

    focal_length = Q[2, 3]
    baseline = abs(1.0 / Q[3, 2]) if Q[3, 2] != 0 else 60.0

    print(f"[INFO] Calibration loaded: {image_w}x{image_h}")
    print(f"       Focal: {focal_length:.1f}px, Baseline: {baseline:.1f}mm")

    # ── Open cameras ─────────────────────────────────────────────────────
    cap_left = cv2.VideoCapture(args.left)
    cap_right = cv2.VideoCapture(args.right)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("[ERROR] Cannot open cameras.")
        sys.exit(1)

    for cap in (cap_left, cap_right):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_h)

    # ── Create tuning window ─────────────────────────────────────────────
    window_name = "StereoSGBM Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Trackbars
    cv2.createTrackbar("numDisparities (x16)", window_name, 8, 16, nothing)   # 128 default
    cv2.createTrackbar("blockSize (odd)", window_name, 4, 10, nothing)        # 9 default
    cv2.createTrackbar("uniquenessRatio", window_name, 10, 50, nothing)
    cv2.createTrackbar("speckleWindowSize", window_name, 100, 300, nothing)
    cv2.createTrackbar("speckleRange", window_name, 32, 100, nothing)
    cv2.createTrackbar("disp12MaxDiff", window_name, 1, 30, nothing)
    cv2.createTrackbar("preFilterCap", window_name, 63, 100, nothing)

    print("\n=== STEREO TUNER ===")
    print("Adjust trackbars to tune disparity quality.")
    print("Press 'q' to quit.\n")

    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()

            if not ret_l or not ret_r:
                continue

            # Rectify
            rect_l = cv2.remap(frame_l, map1x, map1y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map2x, map2y, cv2.INTER_LINEAR)

            gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

            # Read trackbar values
            num_disp = max(1, cv2.getTrackbarPos("numDisparities (x16)", window_name)) * 16
            block_size = cv2.getTrackbarPos("blockSize (odd)", window_name) * 2 + 5
            uniqueness = cv2.getTrackbarPos("uniquenessRatio", window_name)
            speckle_win = cv2.getTrackbarPos("speckleWindowSize", window_name)
            speckle_range = cv2.getTrackbarPos("speckleRange", window_name)
            disp12 = cv2.getTrackbarPos("disp12MaxDiff", window_name)
            pre_filter = cv2.getTrackbarPos("preFilterCap", window_name)

            # Create stereo matcher with current params
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disp,
                blockSize=block_size,
                P1=8 * 3 * block_size ** 2,
                P2=32 * 3 * block_size ** 2,
                disp12MaxDiff=disp12,
                uniquenessRatio=uniqueness,
                speckleWindowSize=speckle_win,
                speckleRange=speckle_range,
                preFilterCap=pre_filter,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

            # Compute disparity
            disparity = stereo.compute(gray_l, gray_r)
            disp_float = disparity.astype(np.float32) / 16.0

            # Convert to depth
            depth_mm = np.zeros_like(disp_float, dtype=np.uint16)
            valid = disp_float > 0
            depth_mm[valid] = (
                (focal_length * baseline) / disp_float[valid]
            ).astype(np.uint16)
            depth_mm[depth_mm > 10000] = 0

            # Disparity color map
            disp_vis = cv2.normalize(disp_float, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = disp_vis.astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

            # Depth color map
            depth_vis = cv2.convertScaleAbs(depth_mm, alpha=(255.0 / 4000.0))
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Draw epipolar lines on rectified pair
            rect_combined = np.hstack([rect_l, rect_r])
            for y in range(0, image_h, 40):
                cv2.line(rect_combined, (0, y), (rect_combined.shape[1], y),
                         (0, 255, 0), 1)

            # Stats overlay
            valid_depths = depth_mm[depth_mm > 0]
            if valid_depths.size > 0:
                min_d = np.min(valid_depths)
                mean_d = np.mean(valid_depths)
                info = f"Min: {min_d}mm | Mean: {mean_d:.0f}mm | Valid: {valid_depths.size/(image_w*image_h)*100:.0f}%"
            else:
                info = "No valid depth data"

            cv2.putText(disp_color, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()
                cv2.putText(disp_color, f"FPS: {fps:.1f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Params overlay
            params_text = f"numDisp={num_disp} block={block_size} uniq={uniqueness}"
            cv2.putText(depth_color, params_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            rect_resized = cv2.resize(rect_combined, (image_w * 2, image_h))
            bottom = np.hstack([disp_color, depth_color])

            # If sizes match, stack
            if rect_resized.shape[1] == bottom.shape[1]:
                output = np.vstack([rect_resized, bottom])
            else:
                output = bottom

            cv2.imshow(window_name, output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        print("[INFO] Tuner closed.")


if __name__ == "__main__":
    main()
