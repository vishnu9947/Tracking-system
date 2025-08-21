import pymongo

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['attendance_system']
camera_collection = db['cameras']

# Sample camera data
camera_data = [
    {
        "camera_id": "camera_1",
        "resolution": [704, 576],
        "stream_link": "rtsp://admin:Admin$1234@192.168.1.104:554/cam/realmonitor?channel=1&subtype=0",
        "rois": {
            "ROI_1": [[281, 109], [380, 80], [564, 170], [470, 247]],
            "ROI_2": [[100, 200], [200, 180], [300, 250], [150, 300]]
        }
    }

]

# Insert camera data into the collection
try:
    result = camera_collection.insert_many(camera_data)
    print(f"Inserted camera data with IDs: {result.inserted_ids}")
except Exception as e:
    print(f"Error inserting camera data: {e}")
