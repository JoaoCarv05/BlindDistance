import cv2
import numpy as np
from openni import openni2
import time

# 1. Initialize using your working project path
project_path = r"C:\Users\LAOB\Desktop\Mário\BlindDistance"
openni2.initialize(project_path)

# 2. Open device and create depth stream
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

# Define points in a grid pattern across the depth frame
points = {}
for y in range(40, 480, 20):
    for x in range(20, 640, 20):
        points[f"POINT-{y}{x}"] = (y, x)

print("Streaming grid depth... Press 'q' to stop.")

try:
    while True:
        # Reset the list for EVERY frame so we don't store old data
        grid_distance = []
        
        # Grab a frame
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        
        # Convert to a 640x480 numpy array
        img = np.frombuffer(frame_data, dtype=np.uint16).reshape(480, 640)
        
        # --- AI DATA PREPARATION ---
        # The most efficient way to get all distances in a NumPy array at once:
        # This takes a 'snapshot' of the grid values and substitutes them every frame
        ai_grid_array = img[40:480:20, 20:640:20]
        # ---------------------------

        # Visualize the depth
        img_vis = cv2.convertScaleAbs(img, alpha=(255.0/4000.0))
        img_vis = cv2.applyColorMap(img_vis, cv2.COLORMAP_JET)

        # Loop through our defined points to measure and draw
        for name, coords in points.items():
            y, x = coords
            distance = img[y, x]
            
            # Add to list for this specific frame
            grid_distance.append(distance)
            
            # Change color to RED if an object is closer than 1 meter (1000mm)
            color = (0, 255, 0) # Green
            if 0 < distance < 1000:
                color = (0, 0, 255) # Red
            
            # Draw circle at the point
            cv2.circle(img_vis, (x, y), 2, color, -1)
        
        print(f"Current grid distances (mm): {grid_distance}")
        time.sleep(0.1)
        # Here is where you would send 'ai_grid_array' to your Local API
        # print(f"Current center distance: {grid_distance[336]}mm") 

        cv2.imshow("BlindDistance - Grid Monitor", img_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Proper cleanup
    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()