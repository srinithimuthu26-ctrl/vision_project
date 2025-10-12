import numpy as np
import cv2
import cv2.aruco as aruco
import paho.mqtt.client as mqtt
import json
import os
import time

# ==================== MQTT CONFIGURATION ====================
# MQTT broker connection details for remote communication
BROKER = "fesv-mqtt.bath.ac.uk"      # University MQTT server
PORT = 31415                         # MQTT port number
USERNAME = "student"                 # Authentication username
PASSWORD = "HousekeepingGlintsStreetwise"  # Authentication password
TOPIC = "Bombsquad"                  # Topic name for publishing marker data

# Create MQTT client and establish connection
client = mqtt.Client(protocol=mqtt.MQTTv311)  # Use MQTT version 3.1.1
client.username_pw_set(USERNAME, PASSWORD)    # Set authentication
client.connect(BROKER, PORT, 60)              # Connect with 60s keepalive
client.loop_start()                           # Start background network loop

print("✅ Connected to MQTT broker.")
print(f"Publishing to topic: {TOPIC}")

# ==================== GLOBAL VARIABLES ====================
# Distance measurement smoothing
dist_buffers = {}          # Store recent distance measurements per marker
BUF_SIZE = 30             # Number of frames to average for stability
scale_factor = 1.0        # Scale correction factor (if needed)

# Reference coordinate system - YOU MUST MEASURE THESE POSITIONS
# These 3 markers define your world coordinate system on the floor
reference_markers = {
    0: (0.0, 0.0),        # Marker 0 at origin (bottom left corner)
    2: (0.275, 0.0),      # Marker 2 at 27.5cm along X-axis (bottom right)
    4: (0.0, 0.29),       # Marker 4 at 29cm along Y-axis (top)
}

# Mapping and tracking variables
detected_positions = {}    # Store world positions of all detected markers
reference_pixels = {}      # Store pixel positions of reference markers for conversion
targets = []              # Store target waypoints set by mouse clicks
robot_trail = []          # Store robot movement history for path visualization

# ==================== COORDINATE TRANSFORMATION FUNCTIONS ====================
def draw_axes_fallback(img, CM, dist, rvec, tvec, length=0.05):
    """
    Draw 3D coordinate axes on detected markers (fallback if cv2.drawFrameAxes unavailable)
    
    Parameters:
    - img: Image to draw on
    - CM: Camera matrix from calibration
    - dist: Distortion coefficients
    - rvec: Rotation vector of marker
    - tvec: Translation vector of marker
    - length: Length of axes lines in meters
    """
    # Define 3D axis points (origin + X,Y,Z directions)
    axis = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    
    # Project 3D axis points to 2D image coordinates
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, CM, dist)
    
    # Convert to integer pixel coordinates
    o = tuple(int(x) for x in imgpts[0].ravel())  # Origin
    x = tuple(int(x) for x in imgpts[1].ravel())  # X-axis end
    y = tuple(int(x) for x in imgpts[2].ravel())  # Y-axis end
    z = tuple(int(x) for x in imgpts[3].ravel())  # Z-axis end
    
    # Draw colored lines: X=red, Y=green, Z=blue
    cv2.line(img, o, x, (0,0,255), 3)    # Red X-axis
    cv2.line(img, o, y, (0,255,0), 3)    # Green Y-axis
    cv2.line(img, o, z, (255,0,0), 3)    # Blue Z-axis

def world_to_pixel(world_pos, reference_markers, reference_pixels):
    """
    Convert world coordinates (meters) to pixel coordinates for screen display
    
    Uses linear transformation between reference markers to scale coordinates
    
    Parameters:
    - world_pos: (x, y) world coordinates in meters
    - reference_markers: Dict of marker_id -> world_position
    - reference_pixels: Dict of marker_id -> pixel_position
    
    Returns: (pixel_x, pixel_y) or None if conversion not possible
    """
    if len(reference_pixels) >= 2:
        ref_ids = list(reference_markers.keys())
        if len(ref_ids) >= 2:
            # Get two reference points for creating transformation
            world_0 = reference_markers[ref_ids[0]]   # First ref marker world pos
            world_1 = reference_markers[ref_ids[1]]   # Second ref marker world pos
            pixel_0 = reference_pixels[ref_ids[0]]    # First ref marker pixel pos
            pixel_1 = reference_pixels[ref_ids[1]]    # Second ref marker pixel pos
            
            # Calculate scaling factor: pixels per meter
            world_dist = np.sqrt((world_1[0]-world_0[0])**2 + (world_1[1]-world_0[1])**2)
            pixel_dist = np.sqrt((pixel_1[0]-pixel_0[0])**2 + (pixel_1[1]-pixel_0[1])**2)
            scale = pixel_dist / world_dist if world_dist > 0 else 1.0
            
            # Transform world position to pixel coordinates
            pixel_x = pixel_0[0] + (world_pos[0] - world_0[0]) * scale
            pixel_y = pixel_0[1] + (world_pos[1] - world_0[1]) * scale
            
            return (int(pixel_x), int(pixel_y))
    return None

def pixel_to_world(pixel_pos, reference_markers, reference_pixels):
    """
    Convert pixel coordinates to world coordinates (inverse of world_to_pixel)
    
    Used when user clicks on screen to set targets
    
    Parameters:
    - pixel_pos: (pixel_x, pixel_y) screen coordinates
    - reference_markers: Dict of marker_id -> world_position
    - reference_pixels: Dict of marker_id -> pixel_position
    
    Returns: (world_x, world_y) in meters or None if conversion not possible
    """
    if len(reference_pixels) >= 2:
        ref_ids = list(reference_markers.keys())
        if len(ref_ids) >= 2:
            world_0 = reference_markers[ref_ids[0]]
            world_1 = reference_markers[ref_ids[1]]
            pixel_0 = reference_pixels[ref_ids[0]]
            pixel_1 = reference_pixels[ref_ids[1]]
            
            # Calculate scaling factor: meters per pixel
            world_dist = np.sqrt((world_1[0]-world_0[0])**2 + (world_1[1]-world_0[1])**2)
            pixel_dist = np.sqrt((pixel_1[0]-pixel_0[0])**2 + (pixel_1[1]-pixel_0[1])**2)
            scale = world_dist / pixel_dist if pixel_dist > 0 else 1.0
            
            # Transform pixel position to world coordinates
            world_x = world_0[0] + (pixel_pos[0] - pixel_0[0]) * scale
            world_y = world_0[1] + (pixel_pos[1] - pixel_0[1]) * scale
            
            return (world_x, world_y)
    return None

def mouse_callback(event, x, y, flags, param):
    """
    Handle mouse clicks to set target waypoints
    
    Left click converts pixel position to world coordinates and stores as target
    
    Parameters:
    - event: Mouse event type
    - x, y: Pixel coordinates of click
    - flags: Additional mouse state flags
    - param: User data (unused)
    """
    global targets, reference_pixels, reference_markers
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        # Convert pixel click to world coordinates
        if len(reference_pixels) >= 2:
            world_pos = pixel_to_world((x, y), reference_markers, reference_pixels)
            if world_pos:
                targets.append(world_pos)  # Add to targets list
                print(f"Target set at pixel ({x}, {y}) = world ({world_pos[0]:.3f}, {world_pos[1]:.3f})")

def draw_targets_and_trail(frame, targets, robot_trail, reference_markers, reference_pixels):
    """
    Draw visual indicators for targets and robot movement trail
    
    Parameters:
    - frame: Image to draw on
    - targets: List of target world positions
    - robot_trail: List of robot world positions over time
    - reference_markers: Reference marker positions
    - reference_pixels: Reference marker pixel positions
    """
    # Draw targets as blue circles with numbers
    for i, target_world in enumerate(targets):
        target_pixel = world_to_pixel(target_world, reference_markers, reference_pixels)
        if target_pixel:
            # Draw target circle
            cv2.circle(frame, target_pixel, 15, (255, 0, 0), 3)  # Blue circle outline
            cv2.circle(frame, target_pixel, 5, (255, 0, 0), -1)   # Blue filled center
            
            # Draw target number label
            cv2.putText(frame, f"T{i+1}", 
                       (target_pixel[0] + 20, target_pixel[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw robot trail as connected yellow lines
    if len(robot_trail) > 1:
        for i in range(len(robot_trail) - 1):
            start_pixel = world_to_pixel(robot_trail[i], reference_markers, reference_pixels)
            end_pixel = world_to_pixel(robot_trail[i + 1], reference_markers, reference_pixels)
            if start_pixel and end_pixel:
                cv2.line(frame, start_pixel, end_pixel, (0, 255, 255), 2)  # Yellow trail

# ==================== CAMERA CALIBRATION LOADING ====================
calib_path = r"C:\Users\Srinithi\Desktop\MECHATRONICS II\vision_project\workdir\Calibration_v1.npz"

# Check if calibration file exists
if not os.path.exists(calib_path):
    print("❌ Calibration file not found:", calib_path)
    exit()

# Load camera calibration data
data = np.load(calib_path)
CM = data['CM']              # Camera matrix (focal length, optical center)
dist_coef = data['dist_coef'] # Distortion coefficients (barrel/pincushion correction)

print("📷 Calibration loaded successfully.")
print("Camera Matrix:\n", CM)
print("Distortion Coefficients:\n", dist_coef)

# ==================== CAMERA AND ARUCO SETUP ====================
# Initialize camera
cap = cv2.VideoCapture(0)                    # Use camera index 0 (change if needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)    # Set high resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initialize ArUco detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # 4x4 marker dictionary
parameters = aruco.DetectorParameters()                         # Default detection params
detector = aruco.ArucoDetector(aruco_dict, parameters)          # Create detector object
marker_length = 0.10  # Physical marker side length in meters (10cm)

# Control variables
last_payload = None       # Prevent duplicate MQTT messages
frame_counter = 0         # Count processed frames
UPDATE_EVERY = 10         # Reduce print frequency (every 10 frames)
mouse_callback_set = False # Flag to set mouse callback only once

print("\n🚀 Running detection... Press 'q' to quit.\n")

# ==================== MAIN DETECTION LOOP ====================
while True:
    # ========== FRAME CAPTURE ==========
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured.")
        break

    frame_counter += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for detection
    
    # ========== ARUCO MARKER DETECTION ==========
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # Draw green outlines around detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate 3D pose of each detected marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, CM, dist_coef)

        # ========== PROCESS EACH DETECTED MARKER ==========
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Draw 3D coordinate axes on marker
            if hasattr(cv2, "drawFrameAxes"):
                cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 0.05)
            else:
                draw_axes_fallback(frame, CM, dist_coef, rvec, tvec, 0.05)

            # Calculate distance from camera to marker
            euclid_m = float(np.linalg.norm(tvec[0]))   # Euclidean distance in meters
            distance_cm = euclid_m * 100 * scale_factor  # Convert to centimeters
            marker_id = int(ids[i][0])                   # Get marker ID number

            # ========== WORLD COORDINATE CALCULATION ==========
            world_pos = None
            
            if marker_id in reference_markers:
                # REFERENCE MARKER: Use predefined world position
                world_pos = reference_markers[marker_id]
                
                # Store pixel position of reference marker for coordinate conversion
                marker_center = np.mean(corners[i][0], axis=0)
                reference_pixels[marker_id] = tuple(map(int, marker_center))
                
                # Print reference marker position (reduced frequency)
                if frame_counter % UPDATE_EVERY == 0:
                    print(f"📍 Reference Marker {marker_id} at world position: ({world_pos[0]:.3f}, {world_pos[1]:.3f})")
            
            else:
                # ROBOT/UNKNOWN MARKER: Calculate position relative to reference markers
                ref_marker_found = None
                ref_tvec = None
                
                # Find a reference marker that's visible in current frame
                for ref_id in reference_markers:
                    if ref_id in [int(id[0]) for id in ids]:
                        ref_idx = [int(id[0]) for id in ids].index(ref_id)
                        ref_marker_found = ref_id
                        ref_tvec = tvecs[ref_idx]
                        break
                
                if ref_marker_found is not None:
                    # Calculate offset from reference marker in camera space
                    offset_x = tvec[0][0] - ref_tvec[0][0]
                    offset_y = tvec[0][1] - ref_tvec[0][1]
                    
                    # Convert to world coordinates using reference marker position
                    ref_world_pos = reference_markers[ref_marker_found]
                    world_pos = (ref_world_pos[0] + offset_x, ref_world_pos[1] + offset_y)
                    
                    # Store robot position for mapping
                    detected_positions[marker_id] = world_pos
                    
                    # Add to robot trail for path visualization
                    if marker_id not in [trail_pos for trail_pos, _ in robot_trail]:
                        robot_trail.append(world_pos)
                        if len(robot_trail) > 50:  # Limit trail length
                            robot_trail.pop(0)
                    
                    # Print robot position (reduced frequency)
                    if frame_counter % UPDATE_EVERY == 0:
                        print(f"🤖 Robot Marker {marker_id} world position: ({world_pos[0]:.3f}, {world_pos[1]:.3f}) [relative to ref {ref_marker_found}]")

            # ========== DISTANCE MEASUREMENT SMOOTHING ==========
            # Store distance measurements in rolling buffer for stability
            buf = dist_buffers.setdefault(marker_id, [])
            buf.append(euclid_m)
            if len(buf) > BUF_SIZE:
                buf.pop(0)  # Remove oldest measurement
            
            # Print statistics when buffer is full
            if len(buf) == BUF_SIZE:
                m = np.mean(buf)
                s = np.std(buf)
                print(f"📊 DEBUG Marker {marker_id} mean={m*100:.1f}cm std={s*100:.1f}cm (n={BUF_SIZE})")

            # ========== DISPLAY TEXT ON FRAME ==========
            display_text = f"ID {marker_id}  Height: {distance_cm:.1f} cm"
            if world_pos:
                display_text += f"  World: ({world_pos[0]:.2f}, {world_pos[1]:.2f})"
            
            cv2.putText(frame, display_text, (10, 40 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # ========== MQTT PUBLISHING ==========
            # Prepare data payload for network transmission
            payload = {
                "id": marker_id,
                "height_cm": round(float(distance_cm), 1),
                "world_x": round(world_pos[0], 3) if world_pos else None,
                "world_y": round(world_pos[1], 3) if world_pos else None,
                "timestamp": time.time()
            }

            # Publish to MQTT broker (avoid duplicates)
            if payload != last_payload:
                client.publish(TOPIC, json.dumps(payload))
                if frame_counter % UPDATE_EVERY == 0:
                    print(f"📡 Published: {payload}")
                last_payload = payload

        # ========== DRAW TARGETS AND ROBOT TRAIL ==========
        if len(reference_pixels) >= 2:
            draw_targets_and_trail(frame, targets, robot_trail, reference_markers, reference_pixels)

    else:
        # ========== NO MARKERS DETECTED ==========
        no_marker_msg = {"status": "no_marker", "timestamp": time.time()}
        client.publish(TOPIC, json.dumps(no_marker_msg))
        if frame_counter % (UPDATE_EVERY * 3) == 0:  # Less frequent printing
            print("⚪ No marker detected.")

    # ========== DISPLAY FRAME AND HANDLE INPUT ==========
    cv2.imshow("ArUco Detection", frame)
    
    # Set mouse callback after window is created (only once)
    if not mouse_callback_set:
        cv2.setMouseCallback("ArUco Detection", mouse_callback)
        mouse_callback_set = True
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

# ==================== CLEANUP ====================
cap.release()                # Release camera
cv2.destroyAllWindows()      # Close all windows
client.loop_stop()           # Stop MQTT background loop
client.disconnect()          # Disconnect from MQTT broker
print("✅ MQTT disconnected. Camera released.")