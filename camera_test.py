import cv2

def nothing(x): pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Edges")
cv2.createTrackbar("min", "Edges", 50, 500, nothing)
cv2.createTrackbar("max", "Edges", 150, 500, nothing)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 1.4)

    minVal = cv2.getTrackbarPos("min", "Edges")
    maxVal = cv2.getTrackbarPos("max", "Edges")

    edges = cv2.Canny(blurred, minVal, maxVal)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


