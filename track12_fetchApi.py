import cv2
import numpy as np
import time
import pymongo
import requests  # New import for making API calls
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['attendance_system']
attendance_collection = db['attendance']
camera_collection = db['cameras']  # Camera and ROI data are in the same collection

# Time constants
CONTINUOUS_RECOGNITION_TIME = 2  # seconds
EXIT_THRESHOLD_TIME = 120  # 2 minutes in seconds

# Store the state of recognized persons
recognized_faces = {}

# Initialize FaceAnalysis with a model and detection size
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], model='arcface_r50_v1')
app.prepare(ctx_id=0, det_size=(640, 640))  # Larger detection size for better accuracy

def load_camera_data():
    """
    Load camera details and ROIs from MongoDB.
    """
    cameras = list(camera_collection.find())
    camera_data = {}
    for cam in cameras:
        camera_data[cam['camera_id']] = {
            "stream_link": cam['stream_link'],
            "resolution": cam['resolution'],
            "rois": cam['rois']  # Directly get ROIs from the same document
        }
    return camera_data

def is_face_in_roi(bbox_center, roi_points):
    bbox_center = (int(bbox_center[0]), int(bbox_center[1]))
    return cv2.pointPolygonTest(np.array(roi_points, dtype=np.int32), bbox_center, False) >= 0

def load_known_faces(employees):
    known_face_embeddings = []
    known_names = []
    employee_id_mapping = {}

    for employee in employees:
        name = employee['name']
        employee_id = employee['id']
        image_paths = employee['image_paths']

        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is not None:  # Check if image loading is successful
                faces = app.get(image)
                for face in faces:
                    known_face_embeddings.append(face.normed_embedding)
                    known_names.append(name)
                    employee_id_mapping[name] = employee_id
            else:
                print(f"Warning: Could not load image at {image_path}")

    return known_face_embeddings, known_names, employee_id_mapping

def log_entry(employee_id, roi_id):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    # Creating the document with employee_id as a key and status as value
    data = {
        employee_id: 1,  # Dynamic key for employee_id and its status
        "employee_id": employee_id,
        "status": 1,  # 1 for entry
        "time_stamp": timestamp,
        "roi_id": roi_id
    }
    
    try:
        attendance_collection.update_one(
            {"employee_id": employee_id, "status": 1},
            {"$set": data},
            upsert=True  # Insert if not present
        )
        print(f"Entry logged for employee ID {employee_id} at {timestamp} in {roi_id}")
    except Exception as e:
        print(f"Failed to log entry for employee ID {employee_id}: {e}")

def log_exit(employee_id, last_seen_time, roi_id):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_seen_time))
    
    # Creating the document with employee_id as a key and status as value
    data = {
        employee_id: 0,  # Dynamic key for employee_id and its status
        "employee_id": employee_id,
        "status": 0,  # 0 for exit
        "time_stamp": timestamp,
        "roi_id": roi_id
    }
    
    try:
        attendance_collection.update_one(
            {"employee_id": employee_id, "status": 0},
            {"$set": data},
            upsert=True  # Insert if not present
        )
        print(f"Exit logged for employee ID {employee_id} at {timestamp} from {roi_id}")
    except Exception as e:
        print(f"Failed to log exit for employee ID {employee_id}: {e}")

def recognize_faces(frame, known_face_embeddings, known_names, employee_id_mapping, rois):
    distance_threshold = 0.4  # Tighter threshold for better accuracy
    faces = app.get(frame)
    current_time = time.time()

    for face in faces:
        distances = np.dot(known_face_embeddings, face.normed_embedding.T)
        max_similarity_index = np.argmax(distances)
        name = "Unknown"
        roi_id = None
        employee_id = None

        if distances[max_similarity_index] > distance_threshold:
            name = known_names[max_similarity_index]
            employee_id = employee_id_mapping.get(name, "Unknown")

        bbox = face.bbox.astype(int)
        left, top, right, bottom = bbox
        bbox_center = ((left + right) // 2, (top + bottom) // 2)

        for roi_id, roi_points in rois.items():
            if is_face_in_roi(bbox_center, roi_points):
                if name != "Unknown":
                    if name not in recognized_faces:
                        recognized_faces[name] = {
                            "entry_time": current_time,
                            "last_seen": current_time,
                            "employee_id": employee_id,
                            "in_roi": False,
                            "roi_id": roi_id
                        }

                    recognized_faces[name]["last_seen"] = current_time

                    if not recognized_faces[name]["in_roi"] and (current_time - recognized_faces[name]["entry_time"] >= CONTINUOUS_RECOGNITION_TIME):
                        recognized_faces[name]["in_roi"] = True
                        log_entry(recognized_faces[name]["employee_id"], roi_id)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                break
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, bottom + 15), font, 0.5, (255, 255, 255), 1)

    # Check for exit condition
    for person in list(recognized_faces.keys()):
        if current_time - recognized_faces[person]["last_seen"] > EXIT_THRESHOLD_TIME:
            if recognized_faces[person]["in_roi"]:
                log_exit(recognized_faces[person]["employee_id"], recognized_faces[person]["last_seen"], recognized_faces[person]["roi_id"])
            del recognized_faces[person]

    return frame

def process_video_dynamic_skipping(video_capture, known_face_embeddings, known_names, employee_id_mapping, rois):
    frame_skip = 1  # Start with no skipping
    max_processing_time = 0.05  # Target processing time per frame (in seconds)
    processing_time_tolerance = 0.02  # Tolerance before adjusting frame skip
    min_skip = 10  # Minimum frame skip value
    max_skip = 20  # Maximum frame skip value
    frame_count = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames based on the current frame_skip value
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (704, 576))  # Adjust based on your camera's resolution

            # Measure processing start time
            start_time = time.time()

            # Process the frame
            future = executor.submit(recognize_faces, frame, known_face_embeddings, known_names, employee_id_mapping, rois)
            processed_frame = future.result()

            # Measure processing end time
            processing_time = time.time() - start_time

            # Adjust frame skip dynamically based on processing time
            if processing_time > max_processing_time + processing_time_tolerance:
                frame_skip = min(frame_skip + 1, max_skip)  # Increase frame skip to reduce load
            elif processing_time < max_processing_time - processing_time_tolerance:
                frame_skip = max(frame_skip - 1, min_skip)  # Decrease frame skip to process more frames

            cv2.imshow('Video', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

def load_config_from_api(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()  # Return the JSON response as a dictionary
    else:
        print(f"Failed to fetch config: {response.status_code} - {response.text}")
        return None

# Load configuration from API
api_url = 'http://your-api-url.com/config'  # Replace with your actual API URL
config = load_config_from_api(api_url)

if config is None:
    print("Could not load configuration. Exiting...")
    exit()

# Load camera data and known faces
camera_data = load_camera_data()
known_face_embeddings, known_names, employee_id_mapping = load_known_faces(config['employees'])

# Open video capture from camera stream
video_capture = cv2.VideoCapture(camera_data['camera_1']['stream_link'])

# Process the video feed
process_video_dynamic_skipping(video_capture, known_face_embeddings, known_names, employee_id_mapping, camera_data['camera_1']['rois'])
