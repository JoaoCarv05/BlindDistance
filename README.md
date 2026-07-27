# BlindDistance

BlindDistance is an assistive technology application powered by **stereo computer vision**, specifically designed to help visually impaired individuals navigate their surroundings safely. It uses **two conventional USB webcams** to compute real-time depth maps via stereo disparity, detecting obstacles, measuring their distances, and identifying potential hazards (like stairs or sudden drops) to provide audible feedback.

## Key Features
* **Stereo Depth Estimation:** Uses two USB webcams with OpenCV's StereoSGBM algorithm to compute depth maps in real-time — no specialized depth sensor required.
* **Real-time Obstacle Detection:** Uses the `ultralytics` YOLOv8 Nano model for detecting and classifying objects in the frame.
* **Spatial Depth Mapping:** Cross-references YOLO detections with the stereo depth map to determine physical distances (in millimeters) to each detected object.
* **Audible Warnings:** Uses `pyttsx3` and `pygame` to provide spoken warnings and proximity beeps based on distance thresholds, ensuring a hands-free, non-visual experience.
* **Floor Drop / Stair Detection:** Uses a moving exponential average of the floor's depth to determine if the ground suddenly drops off ahead (holes, descending stairs).
* **Dataset Recording:** Allows recording annotated depth and color frames locally for training or tuning models.

## Architecture & Components
* **`stereo_calibration.py`**: Standalone calibration tool. Captures chessboard patterns from both cameras to compute intrinsic/extrinsic parameters and rectification maps. Must be run once before using the system.
* **`stereo_camera.py`**: The `StereoCamera` class — captures from two webcams, rectifies images, computes disparity, and converts to a depth map. Drop-in replacement for the legacy `AstraCamera`.
* **`main.py`**: The main entry point integrating stereo depth, AI detection, and the audio alert system in a central event loop.
* **`utils/audio_feedback.py`**: Non-blocking text-to-speech and beep generation via pyttsx3 + pygame.
* **`utils/vision.py`**: YOLOv8 Nano object detection with depth-distance cross-referencing.
* **`data_recorder.py`**: Saves annotated depth/color/grid data for AI training.
* **`debug_floor_drop.py`**: Diagnostic tool for investigating and tuning floor-drop detection.
* **`stereo_tuner.py`**: Real-time StereoSGBM parameter tuning tool with trackbars.

## Hardware Requirements
* **2 USB webcams** (any standard webcam)
* **Rigid mounting bracket** keeping both cameras at a fixed baseline of ~6-7 cm apart
* **Printed chessboard pattern** (8×6 inner corners, 30mm squares) for calibration
* Computer with multi-core CPU
* Speakers or headphones for audio feedback

## Installation

Requires **Python 3.12+**.

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .\.venv\Scripts\Activate.ps1  # Windows
   ```
2. Install the prerequisites:
   ```bash
   pip install -r requirements.txt
   ```
3. *(Optional)* For smoother depth maps, install opencv-contrib:
   ```bash
   pip install opencv-contrib-python
   ```

## Usage

### Step 1: Calibrate the Stereo Cameras
```bash
python stereo_calibration.py --left 0 --right 1
```
- Position the chessboard in front of both cameras
- Press **SPACE** to capture pairs (need at least 15)
- Press **c** to run calibration
- Press **v** to verify rectification (lines should align horizontally)
- Press **q** to quit

This generates `stereo_calibration_data.xml`.

### Step 2: Run the Main System
```bash
python main.py
```

**Controls:**
* `r`: Toggle data recording mode
* `1-5`: Assign label classes to recorded data
* `q`: Quit

### Optional: Tune Depth Parameters
```bash
python stereo_tuner.py
```
Adjust StereoSGBM parameters in real-time with trackbars to optimize depth quality for your specific setup.

## Troubleshooting
* **"Calibration file not found"**: Run `stereo_calibration.py` first.
* **Poor depth quality**: Re-run calibration with more chessboard pairs from varied angles and distances. Use `stereo_tuner.py` to fine-tune parameters.
* **Cameras swapped**: Use `python stereo_calibration.py --swap` to reverse left/right.
* **Low FPS**: Try reducing `numDisparities` or increasing `blockSize` in `stereo_camera.py`. Ensure no other applications are using the cameras.
