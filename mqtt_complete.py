import paho.mqtt.client as mqtt
import json
import time

class VisionSystemMQTT:
    def __init__(self, student_id="your_id"):
        # Connection settings
        self.broker = "fesv-mqtt.bath.ac.uk"
        self.port = 31415
        self.username = "student"
        self.password = "yourpassword"
        self.student_id = student_id
        
        # Create MQTT client
        self.client = mqtt.Client()
        self.client.username_pw_set(self.username, self.password)
        self.connected = False
        
        # Define topics (like mailbox addresses)
        self.topics = {
            # Topics you SEND to (robot receives)
            'send_command': f'robot/{student_id}/command',
            'send_waypoint': f'robot/{student_id}/waypoint',
            'send_path': f'robot/{student_id}/path',
            
            # Topics you LISTEN to (robot sends)
            'receive_status': f'robot/{student_id}/status',
            'receive_position': f'robot/{student_id}/position'
        }
        
        # Set up event handlers
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        # Track robot state
        self.robot_status = "unknown"
        self.robot_position = None
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ Connected to MQTT broker!")
            self.connected = True
            
            # Subscribe to robot's feedback topics
            client.subscribe(self.topics['receive_status'])
            client.subscribe(self.topics['receive_position'])
            print(f"📡 Listening for robot feedback...")
        else:
            print(f"❌ Connection failed: {rc}")
    
    def on_message(self, client, userdata, msg):
        """Handle messages FROM the robot"""
        try:
            topic = msg.topic
            data = json.loads(msg.payload.decode())
            
            if topic == self.topics['receive_status']:
                self.robot_status = data.get('status', 'unknown')
                print(f"🤖 Robot status: {self.robot_status}")
                
            elif topic == self.topics['receive_position']:
                x = data.get('x', 0)
                y = data.get('y', 0)
                self.robot_position = (x, y)
                print(f"📍 Robot at: ({x:.1f}cm, {y:.1f}cm)")
                
        except Exception as e:
            print(f"❌ Error reading robot message: {e}")
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()  # Start background thread
            time.sleep(2)  # Wait for connection
            return self.connected
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def send_simple_command(self, command):
        """Send simple commands: start, stop, emergency_stop, reset"""
        if not self.connected:
            print("❌ Not connected to MQTT!")
            return False
            
        message = {
            'command': command,
            'timestamp': time.time(),
            'from': 'vision_system'
        }
        
        # Convert to JSON and send
        json_message = json.dumps(message)
        result = self.client.publish(self.topics['send_command'], json_message)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"📤 Sent command: {command}")
            return True
        else:
            print(f"❌ Failed to send command: {command}")
            return False
    
    def send_single_waypoint(self, x_cm, y_cm):
        """Send one coordinate for robot to move to"""
        if not self.connected:
            print("❌ Not connected to MQTT!")
            return False
            
        message = {
            'x': float(x_cm),
            'y': float(y_cm),
            'timestamp': time.time(),
            'from': 'vision_system'
        }
        
        json_message = json.dumps(message)
        result = self.client.publish(self.topics['send_waypoint'], json_message)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"📤 Sent waypoint: ({x_cm:.1f}, {y_cm:.1f})")
            return True
        else:
            print(f"❌ Failed to send waypoint")
            return False
    
    def send_complete_path(self, path_points):
        """Send entire path from your A* algorithm"""
        if not self.connected:
            print("❌ Not connected to MQTT!")
            return False
            
        if not path_points:
            print("❌ No path points to send!")
            return False
        
        # Convert path points to the format robot expects
        waypoints = []
        for point in path_points:
            waypoints.append({
                'x': float(point[0] * 100),  # Convert meters to cm
                'y': float(point[1] * 100)   # Convert meters to cm
            })
        
        message = {
            'waypoints': waypoints,
            'total_waypoints': len(waypoints),
            'timestamp': time.time(),
            'from': 'vision_system'
        }
        
        json_message = json.dumps(message)
        result = self.client.publish(self.topics['send_path'], json_message)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"📤 Sent complete path: {len(waypoints)} waypoints")
            for i, wp in enumerate(waypoints):
                print(f"   WP{i+1}: ({wp['x']:.1f}cm, {wp['y']:.1f}cm)")
            return True
        else:
            print(f"❌ Failed to send path")
            return False
    
    def disconnect(self):
        """Clean disconnect"""
        self.client.loop_stop()
        self.client.disconnect()
        print("📡 Disconnected from MQTT")

# Example usage
if __name__ == "__main__":
    print("🧪 Testing MQTT Communication...")
    
    # Create MQTT connection
    mqtt_system = VisionSystemMQTT(student_id="test123")
    
    if mqtt_system.connect():
        print("✅ Ready to send commands!")
        
        # Test 1: Send simple command
        mqtt_system.send_simple_command("test")
        time.sleep(1)
        
        # Test 2: Send single waypoint
        mqtt_system.send_single_waypoint(50.0, 30.0)
        time.sleep(1)
        
        # Test 3: Send complete path (like from your A* algorithm)
        test_path = [(0.1, 0.1), (0.3, 0.2), (0.5, 0.4)]  # In meters
        mqtt_system.send_complete_path(test_path)
        
        print("\n⏳ Listening for robot responses... Press Enter to quit")
        input()
        
        mqtt_system.disconnect()
    else:
        print("❌ Could not connect to MQTT broker")