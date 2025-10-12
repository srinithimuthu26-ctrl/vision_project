#load the necessary libraries
import numpy as np
import cv2
import cv2.aruco as aruco
import os

# ----------------------------------------------------------
# Fallback axis-drawing function (works on any OpenCV build)
# ----------------------------------------------------------
def draw_axes_fallback(img, CM, dist, rvec, tvec, length=0.05):
    # 3-D points for axis lines: origin + X, Y, Z
    axis = np.float32([[0,0,0],
                       [length,0,0],
                       [0,length,0],
                       [0,0,length]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, CM, dist)
    o = tuple(int(x) for x in imgpts[0].ravel())
    x = tuple(int(x) for x in imgpts[1].ravel())
    y = tuple(int(x) for x in imgpts[2].ravel())
    z = tuple(int(x) for x in imgpts[3].ravel())
    cv2.line(img, o, x, (0,0,255), 3)   # X-axis (red)
    cv2.line(img, o, y, (0,255,0), 3)   # Y-axis (green)
    cv2.line(img, o, z, (255,0,0), 3)   # Z-axis (blue))

# Load camera calibration
data_path = r"C:\Users\Srinithi\Desktop\MECHATRONICS II\vision_project\workdir\Calibration_v1.npz"
data = np.load(data_path)
CM = data['CM']
dist_coef = data['dist_coef']

# Open webcam (0 = external camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Prepare ArUco detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# start capturing and processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)
        marker_length = 0.052  # 7 cm marker side in metres
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, CM, dist_coef)

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Try official OpenCV function first
            if hasattr(cv2, "drawFrameAxes"):
                cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 0.05)
            else:
                draw_axes_fallback(frame, CM, dist_coef, rvec, tvec, 0.05)

            
            # Compute distance along camera’s Z-axis (straight distance)
            z_distance = tvec[0][2] * 100  # convert metres → cm
            cv2.putText(frame,
                        f"ID {ids[i][0]}  Height: {z_distance:.1f} cm",
                        (10, 40 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
