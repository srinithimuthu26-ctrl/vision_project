import numpy as np
import cv2
import cv2.aruco as aruco
import heapq
import math
import time
from collections import defaultdict
import paho.mqtt.client as mqtt
import json
from typing import Optional, List, Tuple, Dict, Any

# =================================================================================
# CONFIGURATION
# =================================================================================

MQTT_BROKER = "fesv-mqtt.bath.ac.uk"
MQTT_PORT = 31415
MQTT_USERNAME = "student"
MQTT_PASSWORD = "HousekeepingGlintsStreetwise"

GRID_SIZE = 30
ROBOT_DIAMETER_CM = 19.5
LED_DETECTION_THRESHOLD = 50
MORSE_DOT_DURATION = 0.3
MORSE_DASH_DURATION = 0.9
MORSE_TOLERANCE = 0.15
MARKER_LENGTH = 0.10  # 10 cm for robot, bomb, obstacles
CORNER_MARKER_LENGTH = 0.08  # 8 cm for corner markers

# =================================================================================
# MQTT COMMUNICATION
# =================================================================================

class MQTTCommunicator:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.connected = False
        self.last_connection_attempt = 0
        self.connection_retry_delay = 2
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.keepalive = 60
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print("MQTT connected successfully")
        else:
            self.connected = False
            print(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            print("MQTT disconnected unexpectedly - will auto-reconnect")
        else:
            print("MQTT disconnected gracefully")

    def _on_publish(self, client, userdata, mid):
        pass

    def ensure_connection(self) -> bool:
        if self.connected:
            return True
        current_time = time.time()
        if current_time - self.last_connection_attempt < self.connection_retry_delay:
            return False
        try:
            self.last_connection_attempt = current_time
            self.client.loop_stop()
            result = self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            if result == mqtt.MQTT_ERR_SUCCESS:
                self.client.loop_start()
                timeout = time.time() + 3
                while time.time() < timeout:
                    if self.connected:
                        return True
                    time.sleep(0.1)
        except Exception as e:
            print(f"MQTT connection error: {e}")
        return False

    def disconnect(self):
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False

    def _publish_with_retry(self, topic: str, data: Dict[Any, Any], max_retries: int = 2) -> bool:
        for attempt in range(max_retries):
            if not self.ensure_connection():
                time.sleep(0.5)
                continue
            try:
                message = json.dumps(data, indent=2)
                result = self.client.publish(topic, message)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    return True
                else:
                    self.connected = False
            except Exception as e:
                print(f"MQTT publish error (attempt {attempt + 1}): {e}")
                self.connected = False
        return False

    def send_navigation_waypoints(self, path_waypoints: List[Tuple[float, float]], mission_id: Optional[str] = None) -> bool:
        if not path_waypoints:
            return False
        try:
            waypoint_data = {
                "timestamp": time.time(),
                "mission_id": mission_id or f"mission_{int(time.time())}",
                "total_waypoints": len(path_waypoints),
                "total_distance_cm": self._calculate_path_distance(path_waypoints),
                "waypoints": [
                    {
                        "index": i,
                        "x_cm": round(wp[0] * 100, 1),
                        "y_cm": round(wp[1] * 100, 1),
                        "action": self._get_waypoint_action(i, len(path_waypoints))
                    }
                    for i, wp in enumerate(path_waypoints)
                ]
            }
            success = self._publish_with_retry("Bombsquad/navigation/waypoints", waypoint_data)
            if success:
                print(f"Sent {len(path_waypoints)} waypoints via MQTT")
            return success
        except Exception as e:
            print(f"Error sending waypoints: {e}")
            return False

    def send_morse_code(self, decoded_code: str, bomb_position: Tuple[float, float], numbers_decoded: List[str]) -> bool:
        try:
            morse_data = {
                "timestamp": time.time(),
                "decoded_code": decoded_code,
                "bomb_position": {
                    "x_cm": round(bomb_position[0] * 100, 1),
                    "y_cm": round(bomb_position[1] * 100, 1)
                },
                "decoding_complete": len(decoded_code) == 3,
                "numbers_decoded": numbers_decoded,
                "led_detection_confidence": 95
            }
            success = self._publish_with_retry("Bombsquad/bomb/morse_code", morse_data)
            if success:
                print(f"Sent decoded morse code: {decoded_code}")
            return success
        except Exception as e:
            print(f"Error sending morse code: {e}")
            return False

    def send_object_detection(self, detected_objects: Dict[int, Tuple[float, float]], obstacles: Dict[int, Tuple[float, float]], corners_detected: int) -> bool:
        try:
            detection_data = {
                "timestamp": time.time(),
                "detection_summary": {
                    "robot_detected": 4 in detected_objects,
                    "bomb_detected": 5 in detected_objects,
                    "obstacles_detected": len(obstacles),
                    "coordinate_system_ready": corners_detected == 4
                },
                "objects": {}
            }
            if 4 in detected_objects:
                pos = detected_objects[4]
                if self._is_world_coordinates(pos):
                    detection_data["objects"]["robot"] = {
                        "id": 4,
                        "position": {"x_cm": round(pos[0] * 100, 1), "y_cm": round(pos[1] * 100, 1)},
                        "confidence": 98
                    }
            if 5 in detected_objects:
                pos = detected_objects[5]
                if self._is_world_coordinates(pos):
                    detection_data["objects"]["bomb"] = {
                        "id": 5,
                        "position": {"x_cm": round(pos[0] * 100, 1), "y_cm": round(pos[1] * 100, 1)},
                        "confidence": 95
                    }
            detection_data["objects"]["obstacles"] = [
                {
                    "id": obs_id,
                    "position": {"x_cm": round(pos[0] * 100, 1), "y_cm": round(pos[1] * 100, 1)}
                }
                for obs_id, pos in obstacles.items()
                if self._is_world_coordinates(pos)
            ]
            success = self._publish_with_retry("Bombsquad/detection/objects", detection_data)
            return success
        except Exception as e:
            print(f"Error sending object detection: {e}")
            return False

    def send_emergency_stop(self):
        emergency_data = {
            "timestamp": time.time(),
            "command": "EMERGENCY_STOP",
            "priority": "CRITICAL"
        }
        return self._publish_with_retry("Bombsquad/emergency/stop", emergency_data)

    def _calculate_path_distance(self, waypoints: List[Tuple[float, float]]) -> float:
        if len(waypoints) < 2:
            return 0.0
        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            total_distance += math.sqrt(dx * dx + dy * dy)
        return round(total_distance * 100, 1)

    def _get_waypoint_action(self, index: int, total_waypoints: int) -> str:
        if index == 0:
            return "start"
        elif index == total_waypoints - 1:
            return "bomb_location"
        else:
            return "move"

    def _is_world_coordinates(self, pos: Tuple[float, float]) -> bool:
        return max(abs(pos[0]), abs(pos[1])) < 10

# =================================================================================
# VISION SYSTEM CLASSES
# =================================================================================

detected_corners = {}
perspective_matrix = None
detected_objects = {}
obstacles = {}
current_path = []
morse_decoder = None
led_detector = None

class CoordinateSystem:
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
    def __init__(self, grid_size=GRID_SIZE, robot_diameter_cm=ROBOT_DIAMETER_CM):
        self.grid_size = grid_size
        self.cell_size = 0.9906 / grid_size
        self.robot_radius_cm = robot_diameter_cm / 2
        robot_radius_cells = int(self.robot_radius_cm / (self.cell_size * 100)) + 2
        self.safety_margin = max(3, robot_radius_cells)
        print(f"Grid: {grid_size}x{grid_size}, Cell: {self.cell_size*100:.1f}cm, Safety: {self.safety_margin} cells")

    def world_to_grid(self, world_pos):
        # Accept (x, y) or (x, y, z) tuples
        x, y = world_pos[:2]
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

class RedLEDDetector:
    def __init__(self):
        self.led_history = []
        self.last_state = False
        self.detection_threshold = LED_DETECTION_THRESHOLD
        self.roi_size = 30

    def detect_red_led(self, frame, bomb_center):
        x, y = int(bomb_center[0]), int(bomb_center[1])
        roi_x1 = max(0, x - self.roi_size)
        roi_y1 = max(0, y - self.roi_size)
        roi_x2 = min(frame.shape[1], x + self.roi_size)
        roi_y2 = min(frame.shape[0], y + self.roi_size)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        red_pixels = np.sum(red_mask > 0)
        led_on = red_pixels > self.detection_threshold
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 1)
        status_color = (0, 0, 255) if led_on else (128, 128, 128)
        cv2.circle(frame, (x + 40, y), 8, status_color, -1)
        return led_on

    def update_led_state(self, led_on, timestamp):
        if led_on != self.last_state:
            self.led_history.append((self.last_state, timestamp))
            self.last_state = led_on

class MorseCodeDecoder:
    def __init__(self):
        self.morse_numbers = {
            '-----': '0', '.----': '1', '..---': '2', '...--': '3',
            '....-': '4', '.....': '5', '-....': '6', '--...': '7',
            '---..': '8', '----.': '9'
        }
        self.dot_duration = MORSE_DOT_DURATION
        self.dash_duration = MORSE_DASH_DURATION
        self.tolerance = MORSE_TOLERANCE
        self.digit_gap_duration = 2.1
        self.decoded_numbers = []
        self.current_sequence = []
        self.waiting_for_digit_end = False

    def analyze_timing_sequence(self, led_history):
        if len(led_history) < 2:
            return None
        current_time = time.time()
        if led_history and not self.waiting_for_digit_end:
            last_state, last_timestamp = led_history[-1]
            if not last_state:
                gap_duration = current_time - last_timestamp
                if gap_duration >= self.digit_gap_duration:
                    self.waiting_for_digit_end = True
                    result = self._process_current_sequence(led_history)
                    if result:
                        return result
        return None

    def _process_current_sequence(self, led_history):
        on_durations = []
        for i in range(len(led_history) - 1):
            state, start_time = led_history[i]
            _, end_time = led_history[i + 1]
            duration = end_time - start_time
            if state:
                on_durations.append(duration)
        morse_chars = []
        for duration in on_durations:
            if abs(duration - self.dot_duration) < self.tolerance:
                morse_chars.append('.')
            elif abs(duration - self.dash_duration) < self.tolerance:
                morse_chars.append('-')
        if len(morse_chars) == 5:
            morse_code = ''.join(morse_chars)
            if morse_code in self.morse_numbers:
                number = self.morse_numbers[morse_code]
                self.decoded_numbers.append(number)
                print(f"Decoded digit {len(self.decoded_numbers)}: {morse_code} -> {number}")
                self.current_sequence = []
                self.waiting_for_digit_end = False
                if len(self.decoded_numbers) == 3:
                    code = ''.join(self.decoded_numbers)
                    print(f"COMPLETE 3-DIGIT BOMB CODE: {code}")
                    return code
        return None

    def reset(self):
        self.decoded_numbers = []
        self.current_sequence = []
        self.waiting_for_digit_end = False

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
    for i in range(len(path) - 1):
        start_pixel = world_to_pixel_coordinates(path[i])
        end_pixel = world_to_pixel_coordinates(path[i + 1])
        if start_pixel and end_pixel:
            cv2.line(frame, start_pixel, end_pixel, (0, 255, 255), 3)
    for i, waypoint in enumerate(path):
        pixel_pos = world_to_pixel_coordinates(waypoint)
        if pixel_pos:
            cv2.circle(frame, pixel_pos, 4, (0, 255, 255), -1)

# =================================================================================
# MAIN PROGRAM
# =================================================================================

def main():
    global detected_corners, perspective_matrix, detected_objects, obstacles, current_path
    global morse_decoder, led_detector, coord_system

    coord_system = CoordinateSystem()
    path_planner = OptimizedPathPlanner()
    led_detector = RedLEDDetector()
    morse_decoder = MorseCodeDecoder()
    mqtt_comm = MQTTCommunicator()
    print("Initializing MQTT connection...")
    mqtt_comm.ensure_connection()

    try:
        data = np.load(r"C:\Users\Srinithi\Desktop\MECHATRONICS II\vision_project\workdir\Calibration_v1.npz")
        camera_matrix = data['CM']
        distortion_coeffs = data['dist_coef']
        print("Camera calibration loaded")
    except Exception as e:
        print(f"Camera calibration file not found: {e}")
        camera_matrix = None
        distortion_coeffs = None

    cap = None
    for camera_index in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret, test_frame = cap.read()
                if ret:
                    print(f"Camera {camera_index} initialized successfully")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
        except Exception as e:
            print(f"Camera {camera_index} failed: {e}")
            if cap:
                cap.release()
                cap = None
    if cap is None:
        print("No camera found. Exiting.")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    print("Controls: SPACE=Calculate&Send Path, R=Reset Path, M=Reset Morse, Q=Quit")
    decoded_code = None
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            if camera_matrix is not None and distortion_coeffs is not None:
                frame = cv2.undistort(frame, camera_matrix, distortion_coeffs)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            frame_count += 1
            detected_objects.clear()
            obstacles.clear()
            marker_tvecs = {}  # Store tvecs for all detected markers
            bomb_center = None

            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                for i, marker_id in enumerate(ids.flatten()):
                    # Use correct marker size for pose estimation
                    marker_length = CORNER_MARKER_LENGTH if marker_id in [0, 1, 2, 3] else MARKER_LENGTH
                    if camera_matrix is not None and distortion_coeffs is not None:
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [corners[i]], marker_length, camera_matrix, distortion_coeffs)
                        rvec = rvec[0][0]
                        tvec = tvec[0][0]  # tvec is in meters
                        marker_tvecs[marker_id] = tvec
                        print(f"Marker ID {marker_id}: rvec = {rvec}, tvec = {tvec}")
                        cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, marker_length/2)
                    marker_center = np.mean(corners[i][0], axis=0)
                    center_pixel = tuple(map(int, marker_center))
                    # Use tvec as world coordinates (in meters)
                    world_coords = tuple(tvec) if marker_id in marker_tvecs else None
                    if marker_id in [0, 1, 2, 3]:
                        if world_coords is not None:
                            detected_objects[marker_id] = world_coords
                            detected_corners[marker_id] = world_coords  # Now stores tvec, not pixel
                            cv2.putText(frame, f"C{marker_id}", (center_pixel[0]-20, center_pixel[1]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    elif marker_id == 4:
                        detected_objects[4] = world_coords if world_coords is not None else center_pixel
                        cv2.circle(frame, center_pixel, 15, (0, 0, 255), 2)
                        cv2.putText(frame, f"ROBOT", (center_pixel[0]-30, center_pixel[1]-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif marker_id == 5:
                        detected_objects[5] = world_coords if world_coords is not None else center_pixel
                        bomb_center = center_pixel
                        cv2.circle(frame, center_pixel, 12, (0, 165, 255), 2)
                        cv2.putText(frame, f"BOMB", (center_pixel[0]-25, center_pixel[1]-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        led_on = led_detector.detect_red_led(frame, center_pixel)
                        led_detector.update_led_state(led_on, time.time())
                    elif marker_id >= 6:
                        if world_coords is not None:
                            obstacles[marker_id] = world_coords
                            detected_objects[marker_id] = world_coords
                            cv2.circle(frame, center_pixel, 8, (255, 0, 255), 2)

            # Example: Calculate distances between markers (robot to bomb)
            if 4 in marker_tvecs and 5 in marker_tvecs:
                robot_tvec = marker_tvecs[4]
                bomb_tvec = marker_tvecs[5]
                distance_robot_bomb = np.linalg.norm(robot_tvec - bomb_tvec)  # in meters
                print(f"Distance Robot-Bomb: {distance_robot_bomb*100:.1f} cm")

            # Example: Distance from camera to each marker
            for marker_id, tvec in marker_tvecs.items():
                camera_distance = np.linalg.norm(tvec)  # in meters
                print(f"Distance from camera to marker {marker_id}: {camera_distance*100:.1f} cm")

            if bomb_center and not decoded_code:
                result = morse_decoder.analyze_timing_sequence(led_detector.led_history)
                if result:
                    decoded_code = result
                    if 5 in detected_objects:
                        bomb_pos = detected_objects[5]
                        if mqtt_comm._is_world_coordinates(bomb_pos):
                            mqtt_comm.send_morse_code(decoded_code, bomb_pos, morse_decoder.decoded_numbers)

            if len(detected_corners) == 4 and perspective_matrix is None:
                if calculate_perspective_transform():
                    print("Coordinate system established")

            if current_path:
                draw_path(frame, current_path)

            if frame_count % 180 == 0:
                if mqtt_comm.ensure_connection():
                    mqtt_comm.send_object_detection(detected_objects, obstacles, len(detected_corners))

            status_y = 30
            cv2.putText(frame, f"Corners: {len(detected_corners)}/4 | Obstacles: {len(obstacles)} | MQTT: {'OK' if mqtt_comm.connected else 'NO'}",
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status_y += 25
            robot_text = "Robot: Detected" if 4 in detected_objects else "Robot: Missing"
            bomb_text = "Bomb: Detected" if 5 in detected_objects else "Bomb: Missing"
            cv2.putText(frame, f"{robot_text} | {bomb_text}", (10, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status_y += 25
            path_text = f"Path: {len(current_path)} waypoints" if current_path else "Path: None"
            cv2.putText(frame, path_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status_y += 25
            morse_text = f"Decoded Numbers: {len(morse_decoder.decoded_numbers)}/3"
            if decoded_code:
                morse_text = f"CODE DECODED: {decoded_code}"
            cv2.putText(frame, morse_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE=Path | R=Reset Path | M=Reset Morse | Q=Quit",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Integrated Robot Vision + MQTT System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if 4 in detected_objects and 5 in detected_objects:
                    robot_pos = np.array(detected_objects[4])
                    bomb_pos = np.array(detected_objects[5])
                    if (mqtt_comm._is_world_coordinates(tuple(robot_pos)) and
                        mqtt_comm._is_world_coordinates(tuple(bomb_pos))):
                        obstacle_positions = list(obstacles.values())
                        # Calculate direction and stop position
                        vector_to_bomb = bomb_pos - robot_pos
                        distance = np.linalg.norm(vector_to_bomb)
                        robot_radius = 0.0975  # 9.75 cm in meters
                        bomb_radius = 0.065    # 6.5 cm in meters
                        gap = 0.02             # 2 cm in meters (adjust as needed)
                        stop_distance = robot_radius + bomb_radius + gap
                        if distance > stop_distance:
                            direction = vector_to_bomb / distance
                            stop_pos = bomb_pos - direction * stop_distance
                        else:
                            stop_pos = robot_pos  # Already close enough
                        # Plan path to stop_pos instead of bomb_pos
                        current_path = path_planner.find_path(tuple(robot_pos), tuple(stop_pos), obstacle_positions)
                        if current_path:
                            print(f"Path calculated: {len(current_path)} waypoints")
                            success = mqtt_comm.send_navigation_waypoints(current_path)
                            if success:
                                print("Path sent to robot via MQTT")
                            else:
                                print("Failed to send path via MQTT")
                        else:
                            print("No path found")
                    else:
                        print("Need coordinate system (all 4 corner markers) for path planning")
                else:
                    print("Need robot (ID 4) and bomb (ID 5) markers")
            elif key == ord('r'):
                current_path = []
                print("Path reset")
            elif key == ord('m'):
                morse_decoder.reset()
                led_detector.led_history = []
                decoded_code = None
                print("Morse decoder reset")
            elif key == ord('e'):
                print("Sending emergency stop")
                mqtt_comm.send_emergency_stop()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        mqtt_comm.disconnect()
        print("System shutdown complete")

if __name__ == "__main__":
    main()