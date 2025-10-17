import paho.mqtt.client as mqtt
import json
import time

# MQTT Configuration
broker = "fesv-mqtt.bath.ac.uk"
port = 31415
username = "student"
password = "HousekeepingGlintsStreetwise"

def test_navigation_waypoints():
    """Vision System -> Robot: Send calculated path"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        waypoint_data = {
            "timestamp": time.time(),
            "mission_id": "mission_001",
            "total_waypoints": 5,
            "total_distance_cm": 67.3,
            "waypoints": [
                {"index": 0, "x_cm": 15.2, "y_cm": 20.8, "action": "start"},
                {"index": 1, "x_cm": 25.5, "y_cm": 30.1, "action": "move"},
                {"index": 2, "x_cm": 45.0, "y_cm": 45.0, "action": "move"},
                {"index": 3, "x_cm": 65.2, "y_cm": 55.8, "action": "move"},
                {"index": 4, "x_cm": 75.1, "y_cm": 68.3, "action": "bomb_location"}
            ]
        }
        
        result = client.publish("Bombsquad/navigation/waypoints", json.dumps(waypoint_data, indent=2))
        print(f"✓ Navigation waypoints: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Navigation waypoints error: {e}")

def test_navigation_commands():
    """Vision System -> Robot: Movement commands"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        command_data = {
            "timestamp": time.time(),
            "command": "start_navigation",
            "priority": "normal",
            "mission_id": "mission_001"
        }
        
        result = client.publish("Bombsquad/navigation/commands", json.dumps(command_data, indent=2))
        print(f"✓ Navigation commands: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Navigation commands error: {e}")

def test_robot_position():
    """Robot -> Vision System: Current robot location"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        position_data = {
            "timestamp": time.time(),
            "robot_id": "bomb_defusal_robot_01",
            "position": {
                "x_cm": 32.7,
                "y_cm": 28.4,
                "heading_degrees": 67
            },
            "navigation_status": {
                "current_waypoint_index": 1,
                "target_waypoint": {"x_cm": 45.0, "y_cm": 45.0},
                "distance_to_target_cm": 18.3,
                "movement_state": "moving"
            },
            "system_status": {
                "battery_level": 82,
                "speed_cm_per_sec": 4.5,
                "obstacles_detected": False
            }
        }
        
        result = client.publish("Bombsquad/robot/position", json.dumps(position_data, indent=2))
        print(f"✓ Robot position: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Robot position error: {e}")

def test_robot_feedback():
    """Robot -> Vision System: Responses and confirmations"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        feedback_data = {
            "timestamp": time.time(),
            "feedback_type": "waypoint_reached",
            "waypoint_index": 2,
            "next_action_request": "continue_to_next",
            "estimated_time_to_next_seconds": 15,
            "system_status": "operational"
        }
        
        result = client.publish("Bombsquad/robot/feedback", json.dumps(feedback_data, indent=2))
        print(f"✓ Robot feedback: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Robot feedback error: {e}")

def test_mission_status():
    """Vision System -> Robot: Mission progress updates"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        mission_data = {
            "timestamp": time.time(),
            "mission_status": "navigation_in_progress",
            "bomb_detected": True,
            "bomb_position": {"x_cm": 75.1, "y_cm": 68.3},
            "robot_position": {"x_cm": 32.7, "y_cm": 28.4},
            "obstacles_detected": 3,
            "path_ready": True,
            "completion_percentage": 40
        }
        
        result = client.publish("Bombsquad/mission/status", json.dumps(mission_data, indent=2))
        print(f"✓ Mission status: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Mission status error: {e}")

def test_mission_commands():
    """Vision System -> Robot: High-level mission commands"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        command_data = {
            "timestamp": time.time(),
            "command": "start_mission",
            "mission_type": "bomb_defusal",
            "priority": "high",
            "expected_duration_minutes": 5
        }
        
        result = client.publish("Bombsquad/mission/commands", json.dumps(command_data, indent=2))
        print(f"✓ Mission commands: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Mission commands error: {e}")

def test_object_detection():
    """Vision System -> Robot: Real-time object detection"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        detection_data = {
            "timestamp": time.time(),
            "detection_summary": {
                "robot_detected": True,
                "bomb_detected": True,
                "obstacles_detected": 3,
                "coordinate_system_ready": True
            },
            "objects": {
                "robot": {"id": 4, "position": {"x_cm": 32.7, "y_cm": 28.4}, "confidence": 98},
                "bomb": {"id": 5, "position": {"x_cm": 75.1, "y_cm": 68.3}, "confidence": 95},
                "obstacles": [
                    {"id": 6, "position": {"x_cm": 45.0, "y_cm": 35.0}},
                    {"id": 7, "position": {"x_cm": 60.0, "y_cm": 50.0}},
                    {"id": 8, "position": {"x_cm": 30.0, "y_cm": 65.0}}
                ]
            }
        }
        
        result = client.publish("Bombsquad/detection/objects", json.dumps(detection_data, indent=2))
        print(f"✓ Object detection: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Object detection error: {e}")

def test_bomb_morse_code():
    """Vision System -> Robot: Decoded morse code from bomb LED"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        morse_data = {
            "timestamp": time.time(),
            "decoded_code": "542",
            "bomb_position": {"x_cm": 75.1, "y_cm": 68.3},
            "decoding_complete": True,
            "numbers_decoded": ["5", "4", "2"],
            "led_detection_confidence": 95,
            "decoding_duration_seconds": 12.3
        }
        
        result = client.publish("Bombsquad/bomb/morse_code", json.dumps(morse_data, indent=2))
        print(f"✓ Bomb morse code: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Bomb morse code error: {e}")

def test_bomb_approach():
    """Vision System -> Robot: Bomb approach notifications"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        approach_data = {
            "timestamp": time.time(),
            "alert": "approaching_bomb",
            "distance_to_bomb_cm": 8.5,
            "bomb_position": {"x_cm": 75.1, "y_cm": 68.3},
            "approach_angle_degrees": 180,
            "next_action": "prepare_for_defusal"
        }
        
        result = client.publish("Bombsquad/bomb/approach", json.dumps(approach_data, indent=2))
        print(f"✓ Bomb approach: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Bomb approach error: {e}")

def test_bomb_defusal():
    """Robot -> Vision System: Bomb defusal attempt results"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        defusal_data = {
            "timestamp": time.time(),
            "action": "defusal_attempt",
            "code_entered": "542",
            "defusal_result": "success",
            "attempt_number": 1,
            "time_remaining_seconds": 45
        }
        
        result = client.publish("Bombsquad/bomb/defusal", json.dumps(defusal_data, indent=2))
        print(f"✓ Bomb defusal: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Bomb defusal error: {e}")

def test_system_vision():
    """Vision System -> Robot: Vision system health"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        vision_data = {
            "timestamp": time.time(),
            "camera_active": True,
            "corners_detected": 4,
            "coordinate_system_ready": True,
            "detection_confidence": 95,
            "grid_config": {
                "size": "30x30",
                "cell_size_cm": 3.3,
                "safety_margin_cells": 4,
                "robot_diameter_cm": 19.5
            },
            "fps": 28.5
        }
        
        result = client.publish("Bombsquad/system/vision", json.dumps(vision_data, indent=2))
        print(f"✓ System vision: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ System vision error: {e}")

def test_system_health():
    """Both Systems -> Both Systems: Overall system health"""
    client = mqtt.Client()
    client.username_pw_set(username, password)
    
    try:
        client.connect(broker, port, 60)
        
        health_data = {
            "timestamp": time.time(),
            "system_ready": True,
            "mqtt_connected": True,
            "vision_robot_sync": True,
            "last_communication": time.time(),
            "uptime_seconds": 1234,
            "errors": [],
            "performance": {
                "message_rate_per_second": 2.5,
                "average_response_time_ms": 150
            }
        }
        
        result = client.publish("Bombsquad/system/health", json.dumps(health_data, indent=2))
        print(f"✓ System health: {result.rc == 0}")
        client.disconnect()
        
    except Exception as e:
        print(f"✗ System health error: {e}")

def test_all_topics():
    """Publish all MQTT topics in realistic sequence"""
    print("=== Publishing ALL MQTT Topics ===\n")
    
    print("Navigation Topics:")
    test_navigation_waypoints()
    test_navigation_commands()
    time.sleep(1)
    
    print("\nRobot Status Topics:")
    test_robot_position()
    test_robot_feedback()
    time.sleep(1)
    
    print("\nMission Control Topics:")
    test_mission_status()
    test_mission_commands()
    time.sleep(1)
    
    print("\nDetection Topics:")
    test_object_detection()
    time.sleep(1)
    
    print("\nBomb Topics:")
    test_bomb_morse_code()
    test_bomb_approach()
    test_bomb_defusal()
    time.sleep(1)
    
    print("\nSystem Health Topics:")
    test_system_vision()
    test_system_health()
    
    print("\n=== All 12 Topics Published Successfully ===")

if __name__ == "__main__":
    print("Complete MQTT Topics Test")
    print("1. Navigation topics only")
    print("2. Robot status topics only")
    print("3. Mission control topics only")
    print("4. Detection topics only") 
    print("5. Bomb topics only")
    print("6. System health topics only")
    print("7. Publish ALL 12 topics")
    
    choice = input("Choose test (1-7): ")
    
    if choice == "1":
        test_navigation_waypoints()
        test_navigation_commands()
    elif choice == "2":
        test_robot_position()
        test_robot_feedback()
    elif choice == "3":
        test_mission_status()
        test_mission_commands()
    elif choice == "4":
        test_object_detection()
    elif choice == "5":
        test_bomb_morse_code()
        test_bomb_approach()
        test_bomb_defusal()
    elif choice == "6":
        test_system_vision()
        test_system_health()
    elif choice == "7":
        test_all_topics()
    else:
        print("Testing all topics by default...")
        test_all_topics()