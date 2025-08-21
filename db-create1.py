from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client["attendance_system"]
collection = db["cameras_1"]

# Define multiple camera configurations
camera_details = [
    {
        "camera_id": "camera_1",
        "resolution": [704, 576],
        "stream_link": "rtsp://admin:Admin$1234@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1",
        "rois": {
            "ROI_1": [
                [281, 109],
                [380, 80],
                [564, 170],
                [470, 247]
            ],
            "ROI_2": [
                [100, 200],
                [200, 180],
                [300, 250],
                [150, 300]
            ]
        }
    },
    {
        "camera_id": "camera_2",
        "resolution": [1280, 720],
        "stream_link": "rtsp://admin:Admin$1234@192.168.1.105:554/cam/realmonitor?channel=1&subtype=1",
        "rois": {
            "ROI_1": [
                [150, 100],
                [250, 120],
                [350, 200],
                [200, 220]
            ],
            "ROI_2": [
                [400, 300],
                [500, 350],
                [600, 450],
                [450, 500]
            ]
        }
    },
    {
        "camera_id": "camera_3",
        "resolution": [1920, 1080],
        "stream_link": "rtsp://admin:Admin$1234@192.168.1.106:554/cam/realmonitor?channel=1&subtype=1",
        "rois": {
            "ROI_1": [
                [50, 50],
                [150, 50],
                [150, 150],
                [50, 150]
            ],
            "ROI_2": [
                [300, 300],
                [400, 300],
                [400, 400],
                [300, 400]
            ]
        }
    }
]

# Insert multiple camera details into MongoDB
collection.insert_many(camera_details)
print("Camera details added successfully!")
