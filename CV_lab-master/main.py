import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

video_capture = cv2.VideoCapture("Untitled.mp4")

circle_detector = cv2.SimpleBlobDetector_create()

prev_center = None
prev_time = time.time()

speeds, times = list(), list()

trajectory_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

while True:
    ret, frame = video_capture.read()

    if not ret:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    lower_red = np.array([0, 0, 100])
    upper_red = np.array([10, 10, 255])

    red_mask = cv2.inRange(frame, lower_red, upper_red)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if prev_center is not None:
                delta_time = time.time() - prev_time
                speed = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2) / delta_time

                cv2.putText(frame, f"Speed: {speed:.2f} pps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.line(frame, prev_center, center, (0, 255, 0), 3)

                times.append(time.time())
                speeds.append(speed)

            prev_center = center
            prev_time = time.time()

            cv2.circle(frame, center, radius, (0, 255, 0), 4)  # Отобразить круг на кадре

            cv2.circle(trajectory_canvas, center, 2, (0, 255, 0), 3)

    if prev_center is not None:
        cv2.putText(frame, f"Coordinates: {prev_center}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Tracking Red Circle", frame)
    cv2.imshow("Trajectory", trajectory_canvas)

    key = cv2.waitKey(3)
    if key & 0xFF == ord('q'):
        break

plt.plot(times, speeds)
plt.xlabel("Time (sec)")
plt.ylabel("Speed (pps)")
plt.title("Delta speed")
plt.show()

video_capture.release()
cv2.destroyAllWindows()
