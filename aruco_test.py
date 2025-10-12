import numpy as np
import cv2
import cv2.aruco as aruco
import os

#Load camera calibration data
#data_path = r"C:\Users\Srinithi\Desktop\MECHATRONICS II\vision_project\workdir\Calibration.npz"
data = np.load(data_path)
CM = data['CM']
dist_coef = data['dist_coef']

# Open webcam
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load dictionary and parameters (new API)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Draw markers on the frame
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)

        marker_length = 0.07 

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, CM, dist_coef)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 0.03)

    cv2.imshow("ArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
