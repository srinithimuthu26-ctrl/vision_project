import numpy as np
import cv2
import cv2.aruco as aruco
import heapq
import math
import time
from collections import defaultdict

# Global variables
detected_corners = {}
perspective_matrix = None
detected_objects = {}
obstacles = {}
current_path = []
morse_decoder = None
led_detector = None

class CoordinateSystem:
    def __init__(self, width_meters=0.9906, height_meters=0.9906):  # 39 inches = 0.9906 meters
        self.width = width_meters
        self.height = height_meters
        self.corners = {
            0: (0.0, 0.0),                    # Bottom-left
            1: (width_meters, 0.0),           # Bottom-right
            2: (width_meters, height_meters), # Top-right
            3: (0.0, height_meters),          # Top-left
        }

class OptimizedPathPlanner:
    def __init__(self, grid_size=30, robot_diameter_cm=19.5):
        self.grid_size = grid_size
        self.cell_size = 0.9906 / grid_size
        self.robot_radius_cm = robot_diameter_cm / 2
        
        # Calculate safety margin based on robot size
        robot_radius_cells = int(self.robot_radius_cm / (self.cell_size * 100)) + 2
        self.safety_margin = max(3, robot_radius_cells)
        
        print(f"Grid: {grid_size}x{grid_size}, Cell: {self.cell_size*100:.1f}cm, Safety: {self.safety_margin} cells")
    
    def world_to_grid(self, world_pos):
        x, y = world_pos
        grid_x = max(0, min(self.grid_size-1, int(x / self.cell_size)))
        grid_y = max(0, min(self.grid_size-1, int(y / self.cell_size)))
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos):
        x, y = grid_pos
        world_x = (x + 0.5) * self.cell_size
        world_y = (y + 0.5) * self.cell_size
        return (world_x, world_y)
    
    def find_path(self, start_world, goal_world, obstacle_positions):
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)
        
        # Create obstacle grid with robot-aware safety margins
        obstacle_grid = set()
        for obs_pos in obstacle_positions:
            obs_grid = self.world_to_grid(obs_pos)
            for dx in range(-self.safety_margin, self.safety_margin + 1):
                for dy in range(-self.safety_margin, self.safety_margin + 1):
                    ox, oy = obs_grid[0] + dx, obs_grid[1] + dy
                    if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                        obstacle_grid.add((ox, oy))
        
        # A* pathfinding algorithm
        open_list = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        
        while open_list:
            current = heapq.heappop(open_list)[1]
            
            if current == goal:
                # Reconstruct path
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

class RedLEDDetector:
    def __init__(self):
        self.led_history = []
        self.last_state = False
        self.detection_threshold = 50  # Minimum red intensity
        self.roi_size = 30  # Region of interest size around bomb marker
        
    def detect_red_led(self, frame, bomb_center):
        # Extract region around bomb marker
        x, y = int(bomb_center[0]), int(bomb_center[1])
        roi_x1 = max(0, x - self.roi_size)
        roi_y1 = max(0, y - self.roi_size)
        roi_x2 = min(frame.shape[1], x + self.roi_size)
        roi_y2 = min(frame.shape[0], y + self.roi_size)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Convert to HSV for better red detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define red color range in HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Calculate red pixel count
        red_pixels = np.sum(red_mask > 0)
        led_on = red_pixels > self.detection_threshold
        
        # Draw detection area
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 1)
        
        # Show LED state
        status_color = (0, 0, 255) if led_on else (128, 128, 128)
        cv2.circle(frame, (x + 40, y), 8, status_color, -1)
        
        return led_on
    
    def update_led_state(self, led_on, timestamp):
        # Record state change
        if led_on != self.last_state:
            self.led_history.append((self.last_state, timestamp))
            self.last_state = led_on

class MorseCodeDecoder:
    def __init__(self):
        # Morse code dictionary for numbers
        self.morse_numbers = {
            '-----': '0', '.----': '1', '..---': '2', '...--': '3',
            '....-': '4', '.....': '5', '-....': '6', '--...': '7',
            '---..': '8', '----.': '9'
        }
        
        self.dot_duration = 0.3  # Base duration for dot (seconds)
        self.dash_duration = 0.9  # Dash is 3x dot duration
        self.tolerance = 0.15    # Timing tolerance
        
        self.decoded_numbers = []
        self.current_sequence = []
        
    def analyze_timing_sequence(self, led_history):
        if len(led_history) < 2:
            return
        
        # Calculate on/off durations
        durations = []
        for i in range(len(led_history) - 1):
            state, start_time = led_history[i]
            _, end_time = led_history[i + 1]
            duration = end_time - start_time
            
            # Only process ON durations (LED lit periods)
            if state:
                durations.append(duration)
        
        # Convert durations to dots and dashes
        morse_chars = []
        for duration in durations:
            if abs(duration - self.dot_duration) < self.tolerance:
                morse_chars.append('.')
            elif abs(duration - self.dash_duration) < self.tolerance:
                morse_chars.append('-')
        
        # Try to decode as morse numbers
        if len(morse_chars) == 5:  # Complete morse number
            morse_code = ''.join(morse_chars)
            if morse_code in self.morse_numbers:
                number = self.morse_numbers[morse_code]
                self.decoded_numbers.append(number)
                print(f"Decoded: {morse_code} -> {number}")
                
                # Check if we have 3-digit code
                if len(self.decoded_numbers) == 3:
                    code = ''.join(self.decoded_numbers)
                    print(f"Complete 3-digit code: {code}")
                    return code
        
        return None
    
    def reset(self):
        self.decoded_numbers = []
        self.current_sequence = []

# Core coordinate functions
def calculate_perspective_transform():
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
    if marker_id in coord_system.corners:
        return coord_system.corners[marker_id]
    
    if perspective_matrix is not None:
        pixel_array = np.array([[[float(pixel_position[0]), float(pixel_position[1])]]], dtype=np.float32)
        world_coords = cv2.perspectiveTransform(pixel_array, perspective_matrix)
        return (float(world_coords[0][0][0]), float(world_coords[0][0][1]))
    return None

def world_to_pixel_coordinates(world_pos):
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
    if len(path) < 2:
        return
    
    # Draw path lines
    for i in range(len(path) - 1):
        start_pixel = world_to_pixel_coordinates(path[i])
        end_pixel = world_to_pixel_coordinates(path[i + 1])
        
        if start_pixel and end_pixel:
            cv2.line(frame, start_pixel, end_pixel, (0, 255, 255), 3)
    
    # Draw waypoints
    for i, waypoint in enumerate(path):
        pixel_pos = world_to_pixel_coordinates(waypoint)
        if pixel_pos:
            cv2.circle(frame, pixel_pos, 4, (0, 255, 255), -1)

# Initialize components
coord_system = CoordinateSystem()
path_planner = OptimizedPathPlanner(grid_size=30, robot_diameter_cm=19.5)
led_detector = RedLEDDetector()
morse_decoder = MorseCodeDecoder()

# Load camera calibration
data = np.load(r"C:\Users\Srinithi\Desktop\MECHATRONICS II\vision_project\workdir\Calibration_v1.npz")
camera_matrix = data['CM']
distortion_coeffs = data['dist_coef']

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ArUco detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

print("ArUco Path Planning with Morse Code Decoding")
print("Controls: SPACE=Path, R=Reset Path, M=Reset Morse, Q=Quit")

start_time = time.time()
decoded_code = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    current_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    detected_objects.clear()
    obstacles.clear()
    bomb_center = None
    
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        for i, marker_id in enumerate(ids.flatten()):
            marker_center = np.mean(corners[i][0], axis=0)
            center_pixel = tuple(map(int, marker_center))
            
            world_coords = pixel_to_world_coordinates(center_pixel, marker_id)
            
            if world_coords is not None:
                detected_objects[marker_id] = world_coords
                
                # Corner markers (0-3)
                if marker_id in [0, 1, 2, 3]:
                    detected_corners[marker_id] = center_pixel
                    cv2.putText(frame, f"C{marker_id}", 
                               (center_pixel[0]-20, center_pixel[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Robot (ID 4)
                elif marker_id == 4:
                    cv2.circle(frame, center_pixel, 15, (0, 0, 255), 2)
                    cv2.putText(frame, f"ROBOT", 
                               (center_pixel[0]-30, center_pixel[1]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Bomb with LED detection (ID 5)
                elif marker_id == 5:
                    bomb_center = center_pixel
                    cv2.circle(frame, center_pixel, 12, (0, 165, 255), 2)
                    cv2.putText(frame, f"BOMB", 
                               (center_pixel[0]-25, center_pixel[1]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
                    # Detect red LED on bomb
                    led_on = led_detector.detect_red_led(frame, center_pixel)
                    led_detector.update_led_state(led_on, current_time)
                
                # Obstacles (ID 6+)
                elif marker_id >= 6:
                    obstacles[marker_id] = world_coords
                    cv2.circle(frame, center_pixel, 8, (255, 0, 255), 2)
    
    # Morse code decoding
    if bomb_center and not decoded_code:
        result = morse_decoder.analyze_timing_sequence(led_detector.led_history)
        if result:
            decoded_code = result
    
    # Calculate perspective transform
    if len(detected_corners) == 4 and perspective_matrix is None:
        if calculate_perspective_transform():
            print("Coordinate system established")
    
    # Draw current path
    if current_path:
        draw_path(frame, current_path)
    
    # Status display
    status_y = 30
    
    # System status
    cv2.putText(frame, f"Corners: {len(detected_corners)}/4 | Obstacles: {len(obstacles)}", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Robot and bomb status
    status_y += 25
    robot_text = "Robot: Detected" if 4 in detected_objects else "Robot: Missing"
    bomb_text = "Bomb: Detected" if 5 in detected_objects else "Bomb: Missing"
    cv2.putText(frame, f"{robot_text} | {bomb_text}", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Path status
    status_y += 25
    path_text = f"Path: {len(current_path)} waypoints" if current_path else "Path: None"
    cv2.putText(frame, path_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Morse code status
    status_y += 25
    morse_text = f"Decoded Numbers: {len(morse_decoder.decoded_numbers)}/3"
    if decoded_code:
        morse_text = f"CODE DECODED: {decoded_code}"
    cv2.putText(frame, morse_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Controls
    cv2.putText(frame, "SPACE=Path | R=Reset Path | M=Reset Morse | Q=Quit", 
               (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("ArUco Path Planning + Morse Decoding", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
        
    elif key == ord(' '):  # Calculate path
        if 4 in detected_objects and 5 in detected_objects:
            robot_pos = detected_objects[4]
            bomb_pos = detected_objects[5]
            obstacle_positions = list(obstacles.values())
            
            current_path = path_planner.find_path(robot_pos, bomb_pos, obstacle_positions)
            if current_path:
                print(f"Path calculated: {len(current_path)} waypoints")
            else:
                print("No path found")
        else:
            print("Need robot (ID 4) and bomb (ID 5) markers")
            
    elif key == ord('r'):  # Reset path
        current_path = []
        print("Path reset")
        
    elif key == ord('m'):  # Reset morse decoder
        morse_decoder.reset()
        led_detector.led_history = []
        decoded_code = None
        print("Morse decoder reset")

cap.release()
cv2.destroyAllWindows()