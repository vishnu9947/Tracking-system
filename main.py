import cv2
import os
import time

# Replace with your RTSP URL
#rtsp_url = "rtsp://admin:Admin$1234@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1"


#rtsp_url ="rtsp://admin:Admin$1234@192.168.1.104:554/stream"

# Specify the directory where you want to save the frames
save_directory = "/home/force-4/Tracking_face/pics"

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Open a connection to the camera
cap = cv2.VideoCapture(rtsp_url)

# Check if the connection was successful
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Display the frame
    cv2.imshow("CCTV Feed", frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # Press 's' to save the frame
    if key == ord('s'):
        filename = os.path.join(save_directory, f"frame_{int(time.time())}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    # Press 'q' to quit the program
    if key == ord('q'):
        print("Exiting program...")
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
