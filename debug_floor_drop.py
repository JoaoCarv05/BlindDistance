"""
debug_floor_drop.py — Diagnostic tool to investigate depth sensor behavior
for downward hazards (holes, stairs going down, floor drops).

PURPOSE:
  The main BlindDistance system only warns for CLOSE objects (< 1000mm).
  A hole/staircase going DOWN produces a LARGER depth reading or 0 (no return).
  This script visualizes and logs what the sensor actually reports so we can
  determine the right threshold for void detection.

  Updated for STEREO VISION (two USB webcams) — no longer uses OpenNI/Orbbec.

HOW TO USE:
  1. Run stereo_calibration.py first to generate calibration data
  2. Run this script: python debug_floor_drop.py
  3. Point the camera at flat floor first — note the "Floor Mean" value
  4. Then point it at stairs going down, a ledge, or a hole
  5. Watch the "VOID RATIO" and "Floor Mean" change
  6. Press 's' to take a snapshot (saves depth CSV + screenshot)
  7. Press 'c' to calibrate floor baseline (stores current floor mean as reference)
  8. Press 'q' to quit

OUTPUT:
  - Live annotated view: green = normal floor, red = possible void, blue = no-data
  - Console stats every 0.3s: floor mean, max, void ratio, delta from baseline
  - Snapshots saved to debug_snapshots/ folder
"""

import cv2
import numpy as np
import os
import time

from stereo_camera import StereoCamera

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(PROJECT_PATH, "debug_snapshots")
CALIBRATION_FILE = os.path.join(PROJECT_PATH, "stereo_calibration_data.xml")

# Grid sampling (matching main.py settings)
GRID_STEP_Y = 20
GRID_STEP_X = 20
START_Y, START_X = 40, 20

# Floor region: bottom third of the depth frame (where the floor typically is)
# When tilted slightly downward, the lower rows see the ground ahead
FLOOR_REGION_RATIO = 0.33  # bottom 33% of the frame

# Void detection parameters (these are what we're investigating)
VOID_DEPTH_MULTIPLIER = 1.5   # depth > floor_mean * this = potential void
VOID_RATIO_THRESHOLD = 0.20   # if >20% of floor pixels are "void", warn
NO_RETURN_THRESHOLD = 0       # sensor returns 0 when nothing is in range

# ─── Initialization ──────────────────────────────────────────────────────────
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

cam = StereoCamera(CALIBRATION_FILE, left_id=0, right_id=1)

H, W = cam.depth_h, cam.depth_w
print(f"Depth resolution: {W}x{H}")

# Floor region bounds (in pixel rows)
floor_start_row = int(H * (1.0 - FLOOR_REGION_RATIO))
print(f"Floor region: rows {floor_start_row}–{H} (bottom {FLOOR_REGION_RATIO*100:.0f}%)")

# Calibration state
floor_baseline = None  # Will be set when user presses 'c'
snapshot_count = 0
last_print_time = 0

# ─── Helper functions ────────────────────────────────────────────────────────

def analyze_floor(depth_img):
    """Analyze the floor region of a depth frame.
    
    Returns a dict with:
      - floor_region: the raw depth sub-array for the floor
      - valid_mask: boolean mask of valid (non-zero) pixels
      - valid_depths: 1D array of valid depth values
      - floor_mean, floor_median, floor_min, floor_max: stats
      - void_mask: boolean mask of pixels that look like a drop
      - void_ratio: fraction of floor pixels that look like a void
      - no_return_ratio: fraction of pixels with 0 (no depth return)
    """
    floor_region = depth_img[floor_start_row:, :]
    
    valid_mask = floor_region > 0
    valid_depths = floor_region[valid_mask]
    
    result = {
        'floor_region': floor_region,
        'valid_mask': valid_mask,
        'valid_depths': valid_depths,
        'no_return_count': np.sum(~valid_mask),
        'no_return_ratio': np.sum(~valid_mask) / floor_region.size,
        'total_pixels': floor_region.size,
    }
    
    if valid_depths.size > 0:
        result['floor_mean'] = np.mean(valid_depths)
        result['floor_median'] = np.median(valid_depths)
        result['floor_min'] = np.min(valid_depths)
        result['floor_max'] = np.max(valid_depths)
        result['floor_std'] = np.std(valid_depths)
        
        # Void detection: depth significantly larger than the mean
        threshold = result['floor_mean'] * VOID_DEPTH_MULTIPLIER
        void_mask = floor_region > threshold
        # Also count no-return pixels as potential voids
        void_or_noreturn = void_mask | (~valid_mask)
        
        result['void_mask'] = void_mask
        result['void_or_noreturn'] = void_or_noreturn
        result['void_ratio'] = np.sum(void_mask) / floor_region.size
        result['void_or_noreturn_ratio'] = np.sum(void_or_noreturn) / floor_region.size
        result['void_threshold'] = threshold
    else:
        result['floor_mean'] = 0
        result['floor_median'] = 0
        result['floor_min'] = 0
        result['floor_max'] = 0
        result['floor_std'] = 0
        result['void_mask'] = np.zeros_like(floor_region, dtype=bool)
        result['void_or_noreturn'] = np.ones_like(floor_region, dtype=bool)
        result['void_ratio'] = 1.0
        result['void_or_noreturn_ratio'] = 1.0
        result['void_threshold'] = 0

    return result


def build_debug_view(depth_img, analysis):
    """Build a color-coded visualization of the depth frame with floor analysis."""
    # Normalize depth to 8-bit for visualization
    vis = cv2.convertScaleAbs(depth_img, alpha=(255.0 / 5000.0))
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    
    # Overlay floor region analysis
    fr = analysis['floor_region']
    
    # Create overlay for floor region only
    overlay = vis_color[floor_start_row:, :].copy()
    
    # GREEN tint for valid normal-depth floor pixels
    valid_normal = (fr > 0) & (~analysis['void_mask'])
    overlay[valid_normal] = overlay[valid_normal] * 0.5 + np.array([0, 128, 0], dtype=np.uint8) * 0.5
    
    # RED tint for void pixels (depth >> mean)
    overlay[analysis['void_mask']] = overlay[analysis['void_mask']] * 0.3 + np.array([0, 0, 200], dtype=np.uint8) * 0.7
    
    # BLUE tint for no-return pixels
    no_return = fr == 0
    overlay[no_return] = overlay[no_return] * 0.3 + np.array([200, 0, 0], dtype=np.uint8) * 0.7
    
    vis_color[floor_start_row:, :] = overlay
    
    # Draw floor region boundary line
    cv2.line(vis_color, (0, floor_start_row), (W, floor_start_row), (255, 255, 0), 2)
    cv2.putText(vis_color, "FLOOR REGION", (10, floor_start_row - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw stats on image
    y_text = 25
    def put(text, color=(255, 255, 255)):
        nonlocal y_text
        cv2.putText(vis_color, text, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y_text += 22
    
    put(f"Floor Mean: {analysis['floor_mean']:.0f} mm")
    put(f"Floor Std:  {analysis['floor_std']:.0f} mm")
    put(f"Floor Min:  {analysis['floor_min']:.0f}  Max: {analysis['floor_max']:.0f} mm")
    put(f"Void Ratio: {analysis['void_ratio']*100:.1f}%",
        (0, 0, 255) if analysis['void_ratio'] > VOID_RATIO_THRESHOLD else (0, 255, 0))
    put(f"No-Return:  {analysis['no_return_ratio']*100:.1f}%",
        (200, 0, 0) if analysis['no_return_ratio'] > 0.3 else (200, 200, 200))
    
    if floor_baseline is not None:
        delta = analysis['floor_mean'] - floor_baseline
        delta_color = (0, 0, 255) if abs(delta) > 300 else (0, 255, 0)
        put(f"Baseline: {floor_baseline:.0f} mm  Delta: {delta:+.0f} mm", delta_color)
    else:
        put("Press 'c' to calibrate floor baseline", (0, 255, 255))
    
    # Warning banner
    is_void = analysis['void_ratio'] > VOID_RATIO_THRESHOLD
    is_no_return = analysis['no_return_ratio'] > 0.5
    if is_void or is_no_return:
        banner = "!! POTENTIAL DROP DETECTED !!"
        banner_size, _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        bx = (W - banner_size[0]) // 2
        cv2.rectangle(vis_color, (bx - 10, H - 50), (bx + banner_size[0] + 10, H - 10), (0, 0, 180), -1)
        cv2.putText(vis_color, banner, (bx, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    # Key hints
    hint = "c:calibrate  s:snapshot  q:quit"
    cv2.putText(vis_color, hint, (W - 350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    return vis_color


def save_snapshot(depth_img, vis_img, analysis):
    """Save a depth CSV and screenshot for offline analysis."""
    global snapshot_count
    snapshot_count += 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{SNAPSHOT_DIR}/snap_{ts}_{snapshot_count:03d}"
    
    # Save raw depth as CSV
    csv_path = f"{prefix}_depth.csv"
    np.savetxt(csv_path, depth_img, fmt='%d', delimiter=',')
    
    # Save floor region stats
    stats_path = f"{prefix}_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Floor baseline: {floor_baseline}\n")
        f.write(f"Floor mean: {analysis['floor_mean']:.1f}\n")
        f.write(f"Floor median: {analysis['floor_median']:.1f}\n")
        f.write(f"Floor std: {analysis['floor_std']:.1f}\n")
        f.write(f"Floor min: {analysis['floor_min']:.0f}\n")
        f.write(f"Floor max: {analysis['floor_max']:.0f}\n")
        f.write(f"Void ratio: {analysis['void_ratio']*100:.2f}%\n")
        f.write(f"No-return ratio: {analysis['no_return_ratio']*100:.2f}%\n")
        f.write(f"Void threshold: {analysis['void_threshold']:.0f}\n")
    
    # Save visualization screenshot
    img_path = f"{prefix}_view.png"
    cv2.imwrite(img_path, vis_img)
    
    print(f"[SNAPSHOT #{snapshot_count}] Saved to {prefix}_*")


# ─── Main loop ───────────────────────────────────────────────────────────────
print("\n=== FLOOR DROP DIAGNOSTIC (Stereo Vision) ===")
print("Point camera at flat floor, then at stairs/holes.")
print("GREEN overlay = normal floor | RED = void suspect | BLUE = no depth return")
print("Press 'c' to calibrate baseline, 's' for snapshot, 'q' to quit\n")

try:
    while True:
        depth_img, color_img = cam.get_frames()

        if depth_img is None:
            time.sleep(0.01)
            continue
        
        # Analyze
        analysis = analyze_floor(depth_img)
        
        # Build visualization
        vis = build_debug_view(depth_img, analysis)
        
        # Print stats periodically (not every frame — too noisy)
        now = time.time()
        if now - last_print_time > 0.3:
            last_print_time = now
            baseline_str = f"  delta={analysis['floor_mean'] - floor_baseline:+.0f}" if floor_baseline else ""
            void_flag = " ** VOID **" if analysis['void_ratio'] > VOID_RATIO_THRESHOLD else ""
            print(
                f"[FLOOR] mean={analysis['floor_mean']:6.0f}mm  "
                f"std={analysis['floor_std']:5.0f}  "
                f"min={analysis['floor_min']:5.0f}  max={analysis['floor_max']:5.0f}  "
                f"void={analysis['void_ratio']*100:5.1f}%  "
                f"noreturn={analysis['no_return_ratio']*100:4.1f}%"
                f"{baseline_str}{void_flag}"
            )
        
        cv2.imshow("BlindDistance - Floor Drop Diagnostic", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Calibrate: store current floor mean as baseline
            if analysis['floor_mean'] > 0:
                floor_baseline = analysis['floor_mean']
                print(f"\n>>> BASELINE CALIBRATED: {floor_baseline:.0f} mm <<<\n")
            else:
                print("Cannot calibrate — no valid floor depth readings")
        elif key == ord('s'):
            save_snapshot(depth_img, vis, analysis)

finally:
    print("\nShutting down...")
    cam.stop()
    cv2.destroyAllWindows()
    print("Done.")
