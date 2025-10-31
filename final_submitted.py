# Import required libraries
import numpy as np
import cv2
import cv2.aruco as aruco
import heapq
import math
import time
import paho.mqtt.client as mqtt
import json
import struct
from collections import defaultdict

# ===================== MQTT SETUP =====================

class MQTTCommunicator:
    """
    Handles all MQTT communication between the control centre and the robot.
    Publishes commands, waypoints, and receives feedback.
    """
    def __init__(self, broker, port, username, password):
        # Initialize MQTT client and connect to broker
        self.client = mqtt.Client()
        self.client.username_pw_set(username, password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.connected = False
        self.feedback_queue = []
        self.client.connect(broker, port, 60)
        self.client.loop_start()
        # Subscribe to feedback topics for closed-loop control
        self.client.subscribe("robot/feedback")
        self.client.subscribe("robot/turn_feedback")

    def _on_connect(self, client, userdata, flags, rc):
        # Callback for successful connection
        self.connected = (rc == 0)
        print("MQTT connected" if self.connected else f"MQTT connection failed: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        # Callback for disconnection
        self.connected = False
        print("MQTT disconnected")

    def _on_message(self, client, userdata, msg):
        # Callback for receiving messages
        try:
            payload = json.loads(msg.payload.decode())
            self.feedback_queue.append((msg.topic, payload))
        except Exception as e:
            print(f"MQTT message error: {e}")

    def send_scene_initiated(self):
        # Publish scene initiation message
        self.client.publish("scene/init", json.dumps({"scene": "initiated"}))

    def send_bomb_led(self, color):
        # Publish LED color command to bomb
        self.client.publish("bomb/led", json.dumps({"color": color}))

    def send_path_segments(self, distances, angles):
        # Publish path segments (distances and angles) as binary and string for robot navigation
        distances_bytes = struct.pack(f'>{len(distances)}H', *[int(round(d)) for d in distances])
        angles_bytes = struct.pack(f'>{len(angles)}h', *[int(round(a)) for a in angles])
        self.client.publish("robot/distances", distances_bytes)
        self.client.publish("robot/angles", angles_bytes)
        distances_str = ",".join(str(int(round(d))) for d in distances)
        angles_str = ",".join(str(int(round(a))) for a in angles)
        self.client.publish("Bombsquad/robot/distances", distances_str)
        self.client.publish("Bombsquad/robot/angles", angles_str)

    def send_mission_complete(self):
        # Publish mission complete status
        self.client.publish("mission/complete", json.dumps({"status": "success"}))

    def wait_for_feedback(self, expected_type="move", timeout=10):
        # Wait for feedback from robot with timeout
        start = time.time()
        while time.time() - start < timeout:
            if self.feedback_queue:
                topic, payload = self.feedback_queue.pop(0)
                if expected_type == "move" and topic == "robot/feedback":
                    return payload
                elif expected_type == "turn" and topic == "robot/turn_feedback":
                    return payload
            time.sleep(0.05)
        print("MQTT feedback timeout")
        return None

    def disconnect(self):
        # Cleanly disconnect from MQTT broker
        self.client.loop_stop()
        self.client.disconnect()

# ===================== VISION & PATH PLANNING =====================

# Global variables for detected markers, transformation, and path
detected_corners = {}
perspective_matrix = None
detected_objects = {}
obstacles = {}
current_path = []

class CoordinateSystem:
    """
    Defines the real-world coordinate system based on workspace dimensions and ArUco corners.
    """
    def __init__(self, width_meters=0.9906, height_meters=0.9906):
        self.width = width_meters
        self.height = height_meters
        self.corners = {
            0: (0.0, 0.0),
            1: (width_meters, 0.0),
            2: (width_meters, height_meters),
            3: (0.0, height_meters),
        }

class OptimizedPathPlanner:
    """
    Implements A* path planning on a grid, with safety margin for robot size.
    """
    def __init__(self, grid_size=30, robot_diameter_cm=19.5):
        self.grid_size = grid_size
        self.cell_size = 0.9906 / grid_size
        self.robot_radius_cm = robot_diameter_cm / 2
        robot_radius_cells = int(self.robot_radius_cm / (self.cell_size * 100)) + 2
        self.safety_margin = max(3, robot_radius_cells)
        print(f"Grid: {grid_size}x{grid_size}, Cell: {self.cell_size*100:.1f}cm, Safety: {self.safety_margin} cells")
    
    def world_to_grid(self, world_pos):
        # Convert real-world coordinates to grid indices
        x, y = world_pos
        grid_x = max(0, min(self.grid_size-1, int(x / self.cell_size)))
        grid_y = max(0, min(self.grid_size-1, int(y / self.cell_size)))
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos):
        # Convert grid indices back to real-world coordinates
        x, y = grid_pos
        world_x = (x + 0.5) * self.cell_size
        world_y = (y + 0.5) * self.cell_size
        return (world_x, world_y)
    
    def find_path(self, start_world, goal_world, obstacle_positions):
        # Find shortest path from start to goal using A* algorithm
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)
        obstacle_grid = set()
        for obs_pos in obstacle_positions:
            obs_grid = self.world_to_grid(obs_pos)
            for dx in range(-self.safety_margin, self.safety_margin + 1):
                for dy in range(-self.safety_margin, self.safety_margin + 1):
                    ox, oy = obs_grid[0] + dx, obs_grid[1] + dy
                    if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                        obstacle_grid.add((ox, oy))
        open_list = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        while open_list:
            current = heapq.heappop(open_list)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(current))
                    current = came_from[current]
                path.append(self.grid_to_world(start))
                return path[::-1]
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if (neighbor[0] < 0 or neighbor[0] >= self.grid_size or 
                    neighbor[1] < 0 or neighbor[1] >= self.grid_size or
                    neighbor in obstacle_grid):
                    continue
                tentative_g = g_score[current] + math.sqrt(dx*dx + dy*dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = math.sqrt((neighbor[0]-goal[0])**2 + (neighbor[1]-goal[1])**2)
                    f_score = tentative_g + h_score
                    heapq.heappush(open_list, (f_score, neighbor))
        return []

def calculate_perspective_transform():
    """
    Compute the perspective (homography) matrix from detected corners to real-world coordinates.
    """
    global perspective_matrix
    if len(detected_corners) == 4:
        src_points = [detected_corners[i] for i in [0, 1, 2, 3]]
        dst_points = [[coord_system.corners[i][0], coord_system.corners[i][1]] for i in [0, 1, 2, 3]]
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return True
    return False

def pixel_to_world_coordinates(pixel_position, marker_id=None):
    """
    Convert pixel coordinates to real-world coordinates using the homography matrix.
    """
    if marker_id in coord_system.corners:
        return coord_system.corners[marker_id]
    if perspective_matrix is not None:
        pixel_array = np.array([[[float(pixel_position[0]), float(pixel_position[1])]]], dtype=np.float32)
        world_coords = cv2.perspectiveTransform(pixel_array, perspective_matrix)
        return (float(world_coords[0][0][0]), float(world_coords[0][0][1]))
    return None

def world_to_pixel_coordinates(world_pos):
    """
    Convert real-world coordinates to pixel coordinates for visualization.
    """
    if perspective_matrix is not None:
        try:
            inv_matrix = cv2.invert(perspective_matrix)[1]
            world_array = np.array([[[float(world_pos[0]), float(world_pos[1])]]], dtype=np.float32)
            pixel_coords = cv2.perspectiveTransform(world_array, inv_matrix)
            return (int(pixel_coords[0][0][0]), int(pixel_coords[0][0][1]))
        except:
            return None
    return None

def draw_path(frame, path):
    """
    Draw the planned path and waypoints on the camera frame.
    """
    if len(path) < 2:
        return
    for i in range(len(path) - 1):
        start_pixel = world_to_pixel_coordinates(path[i])
        end_pixel = world_to_pixel_coordinates(path[i + 1])
        if start_pixel and end_pixel:
            cv2.line(frame, start_pixel, end_pixel, (0, 255, 255), 3)
    for i, waypoint in enumerate(path):
        pixel_pos = world_to_pixel_coordinates(waypoint)
        if pixel_pos:
            cv2.circle(frame, pixel_pos, 4, (0, 255, 255), -1)

def merge_segments(segments):
    """
    Merge consecutive path segments with similar angles to reduce unnecessary turns.
    """
    if not segments:
        return []
    merged = []
    current_angle = segments[0]['angle_deg']
    current_distance = segments[0]['distance_cm']
    for seg in segments[1:]:
        if abs(seg['angle_deg'] - current_angle) < 1e-2:
            current_distance += seg['distance_cm']
        else:
            merged.append({'distance_cm': round(current_distance, 2), 'angle_deg': round(current_angle, 2)})
            current_angle = seg['angle_deg']
            current_distance = seg['distance_cm']
    merged.append({'distance_cm': round(current_distance, 2), 'angle_deg': round(current_angle, 2)})
    return merged

def rdp_smooth_path(path, epsilon=0.005):
    """
    Apply Ramer-Douglas-Peucker algorithm to smooth the path.
    """
    def rdp(points, epsilon):
        if len(points) < 3:
            return points
        start, end = np.array(points[0]), np.array(points[-1])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return [points[0], points[-1]]
        max_dist = -1
        index = -1
        for i in range(1, len(points)-1):
            pt = np.array(points[i])
            proj = np.dot(pt - start, line_vec) / line_len
            closest = start + proj * line_vec / line_len
            dist = np.linalg.norm(pt - closest)
            if dist > max_dist:
                max_dist = dist
                index = i
        if max_dist > epsilon:
            left = rdp(points[:index+1], epsilon)
            right = rdp(points[index:], epsilon)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]
    return rdp(path, epsilon)

# --- LED Detector Class for Red and Green ---
class LEDDetector:
    """
    Detects the color of the LED in the bomb region using HSV color segmentation.
    """
    def __init__(self, roi_size=30, detection_threshold=50):
        self.roi_size = roi_size
        self.detection_threshold = detection_threshold

    def detect_led(self, frame, bomb_center):
        # Detect if the LED is red or green in the region of interest
        if bomb_center is None:
            return None
        x, y = int(bomb_center[0]), int(bomb_center[1])
        roi_x1 = max(0, x - self.roi_size)
        roi_y1 = max(0, y - self.roi_size)
        roi_x2 = min(frame.shape[1], x + self.roi_size)
        roi_y2 = min(frame.shape[0], y + self.roi_size)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Green range
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.sum(green_mask > 0)
        # Red range (two ranges for hue wraparound)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)
        # Decide which color is dominant
        if green_pixels > self.detection_threshold and green_pixels > red_pixels:
            return 'green'
        elif red_pixels > self.detection_threshold and red_pixels > green_pixels:
            return 'red'
        else:
            return None

# ===================== MAIN =====================

# Initialize coordinate system and path planner
coord_system = CoordinateSystem()
path_planner = OptimizedPathPlanner(grid_size=30, robot_diameter_cm=19.5)

# Load camera calibration data
data = np.load(r"C:\\Users\\Srinithi\\Desktop\\MECHATRONICS II\\vision_project\\workdir\\Calibration_v1.npz")
camera_matrix = data['CM']
distortion_coeffs = data['dist_coef']

# Initialize camera capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set up ArUco marker detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
# ...set other parameters as needed
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Initialize MQTT communication
mqtt_comm = MQTTCommunicator(
    broker="fesv-mqtt.bath.ac.uk",
    port=31415,
    username="student",
    password="HousekeepingGlintsStreetwise"
)

# Print user instructions
print("ArUco Path Planning")
print("Controls: SPACE=Path, R=Reset Path, Q=Quit")

# State variables
scene_initiated = False
mission_complete = False
bomb_center = None
robot_world_coords = None  # (x, y, theta)
bomb_world_coords = None

def get_warped_workspace(frame, detected_corners, output_size=(600, 600)):
    """
    Warp the workspace to a top-down view using the detected ArUco corners.
    """
    if len(detected_corners) != 4:
        return None  # Not all corners detected
    # Order: top-left(0), top-right(1), bottom-right(2), bottom-left(3)
    src_pts = np.array([detected_corners[i] for i in [0, 1, 2, 3]], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, output_size)
    return warped

while True:
    # Main processing loop: capture frame, detect markers, plan path, and communicate
    ret, frame = cap.read()
    if not ret:
        continue
    # Undistort the frame using camera calibration
    frame = cv2.undistort(frame, camera_matrix, distortion_coeffs)
    # Convert frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect ArUco markers in the frame
    corners, ids, _ = detector.detectMarkers(gray)
    # Reset detected objects for this frame
    detected_objects.clear()
    obstacles.clear()
    bomb_center = None
    robot_world_coords = None
    bomb_world_coords = None

    if ids is not None:
        # Process each detected marker
        aruco.drawDetectedMarkers(frame, corners, ids)
        # Estimate pose for each marker
        marker_length = 0.10  # meters (adjust if needed)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, distortion_coeffs)
        for i, marker_id in enumerate(ids.flatten()):
            # Calculate marker center and world coordinates
            marker_center = np.mean(corners[i][0], axis=0)
            center_pixel = tuple(map(int, marker_center))
            world_coords = pixel_to_world_coordinates(center_pixel, marker_id)
            if marker_id in [0, 1, 2, 3]:
                # Corner markers: used for workspace definition
                if world_coords is not None:
                    detected_objects[marker_id] = world_coords
                    detected_corners[marker_id] = center_pixel
                    cv2.putText(frame, f"C{marker_id}", 
                               (center_pixel[0]-20, center_pixel[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif marker_id == 4:
                # Robot marker: estimate position and orientation
                detected_objects[4] = world_coords if world_coords is not None else center_pixel
                # --- Orientation estimation ---
                theta_deg = None
                if rvecs is not None and len(rvecs) > i:
                    rvec = rvecs[i].reshape(3)
                    R, _ = cv2.Rodrigues(rvec)
                    theta_rad = math.atan2(R[1,0], R[0,0])
                    theta_deg = math.degrees(theta_rad)
                if world_coords is not None and theta_deg is not None:
                    robot_world_coords = (world_coords[0], world_coords[1], theta_deg)
                elif world_coords is not None:
                    robot_world_coords = (world_coords[0], world_coords[1], None)
                cv2.circle(frame, center_pixel, 15, (0, 0, 255), 2)
                cv2.putText(frame, f"ROBOT", 
                           (center_pixel[0]-30, center_pixel[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if rvecs is not None and tvecs is not None and len(rvecs) > i and len(tvecs) > i:
                    cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvecs[i], tvecs[i], 0.07)
            elif marker_id == 5:
                # Bomb marker: estimate position
                detected_objects[5] = world_coords if world_coords is not None else center_pixel
                bomb_world_coords = world_coords
                bomb_center = center_pixel
                cv2.circle(frame, center_pixel, 12, (0, 165, 255), 2)
                cv2.putText(frame, f"BOMB", 
                           (center_pixel[0]-25, center_pixel[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            elif marker_id >= 6:
                # Obstacle markers
                if world_coords is not None:
                    obstacles[marker_id] = world_coords
                    detected_objects[marker_id] = world_coords
                    cv2.circle(frame, center_pixel, 8, (255, 0, 255), 2)

    # Show top-down warped workspace
    warped = get_warped_workspace(frame, detected_corners)
    if warped is not None:
        cv2.imshow("Workspace Only (Top-Down)", warped)

    # Calculate perspective transform if all corners are detected
    if len(detected_corners) == 4 and perspective_matrix is None:
        if calculate_perspective_transform():
            print("Coordinate system established")

    # Draw the planned path if available
    if current_path:
        draw_path(frame, current_path)

    # Display status information on the frame
    status_y = 30
    cv2.putText(frame, f"Corners: {len(detected_corners)}/4 | Obstacles: {len(obstacles)}", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    status_y += 25
    robot_text = "Robot: Detected" if robot_world_coords is not None else "Robot: Missing"
    bomb_text = "Bomb: Detected" if bomb_world_coords is not None else "Bomb: Missing"
    cv2.putText(frame, f"{robot_text} | {bomb_text}", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    status_y += 25
    path_text = f"Path: {len(current_path)} waypoints" if current_path else "Path: None"
    cv2.putText(frame, path_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    status_y += 25

    if robot_world_coords is not None:
        x_cm = robot_world_coords[0]*100
        y_cm = robot_world_coords[1]*100
        theta = robot_world_coords[2]
        if theta is not None:
            cv2.putText(frame, f"Robot (cm): x={x_cm:.1f}, y={y_cm:.1f}, theta={theta:.1f} deg",
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            cv2.putText(frame, f"Robot (cm): x={x_cm:.1f}, y={y_cm:.1f}, theta=N/A",
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        status_y += 25
    if bomb_world_coords is not None:
        cv2.putText(frame, f"Bomb (cm): x={bomb_world_coords[0]*100:.1f}, y={bomb_world_coords[1]*100:.1f}",
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        status_y += 25
       
    # Show user controls on the frame
    cv2.putText(frame, "SPACE=Path | R=Reset Path | Q=Quit", 
               (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("ArUco Path Planning", frame)

    # Handle keyboard input for manual control and path planning
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Quit the program
        break
    elif key == 82:  # Up arrow
        # Send forward command
        mqtt_comm.client.publish("robot/control", json.dumps({"command": 1}))
        print("Sent: 1 (forward)")
    elif key == 84:  # Down arrow
        # Send backward command
        mqtt_comm.client.publish("robot/control", json.dumps({"command": 2}))
        print("Sent: 2 (backward)")
    elif key == 81:  # Left arrow
        # Send left command
        mqtt_comm.client.publish("robot/control", json.dumps({"command": 3}))
        print("Sent: 3 (left)")
    elif key == 83:  # Right arrow
        # Send right command
        mqtt_comm.client.publish("robot/control", json.dumps({"command": 4}))
        print("Sent: 4 (right)")
    elif key == ord('s') or key == ord('S'):
        # Send stop command
        mqtt_comm.client.publish("robot/control", json.dumps({"command": 5}))
        print("Sent: 5 (stop)")
    elif key == ord(' '):
        # Plan path and send waypoints if robot and bomb are detected
        if robot_world_coords is not None and bomb_world_coords is not None:
            # Initiate scene and set LED to red if not already done
            if not scene_initiated:
                mqtt_comm.send_scene_initiated()
                print("Scene initiated signal sent.")
                time.sleep(1)
                mqtt_comm.send_bomb_led("red")
                print("Bomb LED set to RED.")
                scene_initiated = True
                time.sleep(1)
            # Compute path to bomb, smooth it, and send to robot
            obstacle_positions = list(obstacles.values())
            robot_pos = np.array(robot_world_coords[:2])
            bomb_pos = np.array(bomb_world_coords)
            vector_to_bomb = bomb_pos - robot_pos
            distance = np.linalg.norm(vector_to_bomb)
            stop_distance = 0.20
            if distance > stop_distance:
                direction = vector_to_bomb / distance
                stop_pos = bomb_pos - direction * stop_distance
            else:
                stop_pos = robot_pos
            raw_path = path_planner.find_path(tuple(robot_pos), tuple(stop_pos), obstacle_positions)
            # Ensure the path starts at the robot's current position
            if raw_path and (np.linalg.norm(np.array(raw_path[0]) - robot_pos) > 1e-6):
                raw_path = [tuple(robot_pos)] + raw_path
            if raw_path:
                print(f"Raw path: {len(raw_path)} waypoints")
                smoothed_path = rdp_smooth_path(raw_path, epsilon=0.01)
                current_path = smoothed_path
                print(f"Smoothed path: {len(current_path)} waypoints")
                raw_segments = []
                for i in range(len(current_path) - 1):
                    x1, y1 = current_path[i]
                    x2, y2 = current_path[i+1]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance_cm = math.sqrt(dx**2 + dy**2) * 100
                    angle_deg = math.degrees(math.atan2(dy, dx))
                    raw_segments.append({'distance_cm': distance_cm, 'angle_deg': angle_deg})
                optimized_segments = merge_segments(raw_segments)
                distances = [seg['distance_cm'] for seg in optimized_segments]
                angles = [seg['angle_deg'] for seg in optimized_segments]
                # Print distances and angles in terminal
                print("Distances (cm):", distances)
                print("Angles (deg):", angles)
                mqtt_comm.send_path_segments(distances, angles)
                print("Path segments sent to robot (as bytes, uint16 for distances, int16 for angles).")
            else:
                print("No path found")
        else:
            print("Need robot (ID 4) and bomb (ID 5) markers")
    elif key == ord('r'):
        # Reset the current path
        current_path = []
        print("Path reset")

# Release resources and disconnect
cap.release()
cv2.destroyAllWindows()
mqtt_comm.disconnect()