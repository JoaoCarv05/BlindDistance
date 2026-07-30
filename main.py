import cv2
import numpy as np
import os
import time

from stereo_camera import StereoCamera
from utils.audio_feedback import AudioFeedback
from utils.vision import ObstacleDetector
from data_recorder import DataRecorder

def draw_osd(img, label, recording, frame_count):
    h, w = img.shape[:2]

    label_text = f"[LABEL: {label}]"
    cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if recording:
        rec_text = "  REC"
        rec_color = (0, 0, 255)
    else:
        rec_text = "  PAUSED"
        rec_color = (180, 180, 180)
    text_size, _ = cv2.getTextSize(rec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(img, rec_text, (w - text_size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)


    count_text = f"Saved: {frame_count} frames"
    cv2.putText(img, count_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


    hint = "1-5:label  r:rec  q:quit"
    hint_size, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(img, hint, (w - hint_size[0] - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

def main():
    project_path = os.path.dirname(os.path.abspath(__file__))
    

    cam = StereoCamera(
        calibration_file=os.path.join(project_path, 'stereo_calibration_data.xml'),
        left_id=0,
        right_id=1
    )
    audio = AudioFeedback()
    vision = ObstacleDetector()
    recorder = DataRecorder(base_dir='dataset', save_fps=2.0)

    # Recording state
    recording = False
    label_idx = 0  # default label: 'clear'
    
    # Grid Settings (dynamic based on camera, avoiding edges)
    grid_step_y = 20
    grid_step_x = 20
   
    start_y, end_y = 40, cam.depth_h
    start_x, end_x = 20, cam.depth_w
    
    # Pre-calculate points for drawing to save loop time
    draw_points = [(y, x) for y in range(start_y, end_y, grid_step_y)
                          for x in range(start_x, end_x, grid_step_x)]


    floor_row_start = int(cam.depth_h * 0.67) 
    floor_void_multiplier = 1.5     
    floor_void_ratio_thresh = 0.25  
    floor_ema_alpha = 0.05          
    floor_baseline = None          


    try:
        while True:
            depth_img, color_img = cam.get_frames()
            
            if depth_img is None or color_img is None:
                time.sleep(0.01)
                continue
            
            ai_grid_matrix = depth_img[start_y:end_y:grid_step_y, start_x:end_x:grid_step_x]
            
            # Flatten to 1D array if needed by external APIs
            grid_distance_flat = ai_grid_matrix.flatten().tolist()
            

            # --- PROXIMITY ALERTS ---
            valid_depths = ai_grid_matrix[ai_grid_matrix > 0]
            if valid_depths.size > 0:
                min_dist = np.min(valid_depths)
                if min_dist < 500:
                    audio.beep(frequency=600, duration_ms=80)  # Soft tone

            # --- FLOOR-DROP DETECTION ---
            floor_region = depth_img[floor_row_start:, :]
            floor_valid = floor_region[(floor_region > 0) & (floor_region < 6000)]

            if floor_valid.size > 0:
                floor_current_median = float(np.median(floor_valid))


                if floor_baseline is None:
                    floor_baseline = floor_current_median
                else:
                    floor_baseline = (floor_ema_alpha * floor_current_median
                                      + (1 - floor_ema_alpha) * floor_baseline)


                void_threshold = floor_baseline * floor_void_multiplier
                void_pixels = np.sum(floor_region > void_threshold)
 
                no_return_pixels = np.sum(floor_region == 0)
                suspicious_ratio = (void_pixels + no_return_pixels) / floor_region.size

                if suspicious_ratio > floor_void_ratio_thresh:
                    audio.speak("Atenção! Desnível à frente.", force=False)

            # 2. RUN VISION AI 

            annotated_img, threats = vision.process_frame(color_img, depth_img)
            
            # 3. CONTEXTUAL AUDIO WARNINGS
            for threat in threats:
                dist_mm = threat['distance_mm']
                name = threat['label']

                if dist_mm == 0:
                    audio.speak(
                        f"Atenção! {name} muito perto.",
                        cooldown_key=name
                    )
                elif dist_mm < 500:
                    audio.speak(
                        f"Cuidado! {name} a {dist_mm/1000:.1f} metros.",
                        cooldown_key=name
                    )
                elif dist_mm < 1000:
                    audio.speak(
                        f"{name} a {dist_mm/1000:.1f} metros.",
                        cooldown_key=name
                    )
            

            for y, x in draw_points:
                dist = depth_img[y, x]
                color = (0, 0, 255) if (0 < dist < 500) else (0, 255, 0)
                cv2.circle(annotated_img, (x, y), 2, color, -1)

            # 5. SAVE TRAINING DATA
            current_label = DataRecorder.LABELS[label_idx]
            if recording:
                recorder.save_frame(depth_img, annotated_img, grid_distance_flat, current_label)

            # 6. OSD OVERLAY
            draw_osd(annotated_img, current_label, recording,
                     recorder.get_frame_count(current_label))

            # 7. UI DISPLAY
            cv2.imshow("BlindDistance - AI Augmented View", annotated_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                state = "STARTED" if recording else "PAUSED"
                print(f"Recording {state} — label: {DataRecorder.LABELS[label_idx]}")
            elif key in DataRecorder.LABEL_KEYS:
                label_idx = DataRecorder.LABEL_KEYS[key]
                print(f"Label changed to: {DataRecorder.LABELS[label_idx]}")
                
    finally:
        print("Shutting down...")
        audio.stop()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
