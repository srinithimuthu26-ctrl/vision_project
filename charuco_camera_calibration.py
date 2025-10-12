import numpy as np
import cv2
import cv2.aruco as aruco


charuco_width = 7
charuco_height = 5
square_length = 0.04


aruco_dict = aruco.Dictionary_create(50, 4)
parameters = aruco.DetectorParameters_create()


charuco_board = aruco.CharucoBoard_create(charuco_width, charuco_height, square_length, 0.8 * square_length, aruco_dict)


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)


        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
        if retval > 0:
            aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)


    cv2.imshow('frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


all_charuco_corners = []
all_charuco_ids = []


for i in range(50):  # Process for 50 frames
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
        if retval > 0:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


calibration_flags = cv2.CALIB_RATIONAL_MODEL
ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners, all_charuco_ids, charuco_board, gray.shape[::-1], None, None, flags=calibration_flags)


print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)