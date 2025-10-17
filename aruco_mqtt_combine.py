import numpy as np
import cv2
import cv2.aruco as aruco

def print_corner_tvecs(rvecs, tvecs, ids, frame_counter):
    """Print tvec coordinates for corner markers (IDs 0,1,2,3)"""
    # Only print every 90 frames to avoid spam
    if frame_counter % 90 != 0:
        return
    
    print("\n=== CORNER MARKER TVEC COORDINATES ===")
    
    corner_found = False
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in [0, 1, 2, 3]:  # Corner markers
            corner_found = True
            tvec = tvecs[i][0]
            distance = np.linalg.norm(tvec)
            
            print(f"Corner Marker {marker_id}:")
            print(f"  tvec: [{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}] meters")
            print(f"  Distance: {distance:.4f}m ({distance*100:.1f}cm)")
            print()
    
    if not corner_found:
        print("No corner markers (IDs 0,1,2,3) detected")
    print("=" * 45)

# Load calibration (must exist)
calib_path = r"C:\Users\Srinithi\Desktop\MECHATRONICS II\vision_project\workdir\Calibration_v1.npz"

try:
    data = np.load(calib_path)
    CM = data['CM']
    dist_coef = data['dist_coef']
    print("Calibration loaded successfully")
except FileNotFoundError:
    print(f"Error: Calibration file not found at {calib_path}")
    print("Please ensure the calibration file exists before running.")
    exit(1)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.128  # 12.8cm markers

frame_counter = 0

print("ArUco tvec Detection System")
print("Place corner markers (IDs 0,1,2,3) to see their positions")
print("Press 'q' to quit, 't' to print tvec immediately")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, CM, dist_coef)

            # Print tvec coordinates
            print_corner_tvecs(rvecs, tvecs, ids, frame_counter)

            # Draw axis for each marker
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 0.05)
                
                # Display basic info on frame
                marker_id = int(ids[i][0])
                distance = np.linalg.norm(tvec[0]) * 100  # Convert to cm
                text = f"ID {marker_id}: {distance:.1f}cm"
                cv2.putText(frame, text, (10, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show detection count
        detected_count = len(ids) if ids is not None else 0
        status = f"Detected markers: {detected_count}"
        cv2.putText(frame, status, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ArUco tvec Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):  # Manual print
            if ids is not None and len(ids) > 0:
                print_corner_tvecs(rvecs, tvecs, ids, 0)
            else:
                print("No markers detected")

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("System shutdown complete")