import cv2
from pymongo import MongoClient
import numpy as np

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
database = client['Productivity_App_Test']
collection = database['Coordinates_Tester']

coordinates = []

def select_coordinates(event, x, y, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(coordinates) < 4:
            coordinates.append((x, y))
            print(f"Selected coordinate {len(coordinates)}: ({x}, {y})")
            if len(coordinates) == 4:
                seat_name = input("Enter the name of the seat: ")
                document = {"Seat": seat_name, "Coordinates": coordinates}
                collection.insert_one(document)
                print("Coordinates saved to MongoDB.")
        else:
            print("You have already selected 4 coordinates.")

def draw_polygon(frame, coordinates):
    if len(coordinates) == 4:
        cv2.polylines(frame, [np.array(coordinates, np.int32)], True, (0, 0, 255), 2)

cv2.namedWindow('Select Coordinates')
cv2.setMouseCallback('Select Coordinates', select_coordinates)

cap = cv2.VideoCapture("rtsp://admin:Admin$1234@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame (width, height)
    frame = cv2.resize(frame, (640, 640))  # Change (640, 480) to desired size
    
    draw_polygon(frame, coordinates)
    cv2.imshow("Select Coordinates", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("e"):
        print("Enter the new coordinates:")
        coordinates = []

cap.release()
cv2.destroyAllWindows()
