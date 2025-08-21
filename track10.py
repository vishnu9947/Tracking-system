import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor
import pymongo

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['attendance_system']
attendance_collection = db['attendance']

# Time constants
CONTINUOUS_RECOGNITION_TIME = 2  # seconds
EXIT_THRESHOLD_TIME = 120  # 2 minutes in seconds

# Store the state of recognized persons
recognized_faces = {}

# Initialize FaceAnalysis with a model and detection size
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], model='arcface_r50_v1')
app.prepare(ctx_id=0, det_size=(640, 640))  # Larger detection size for better accuracy

# Define multiple ROIs with unique IDs
ROIS = {
    "ROI_1": [[281, 109], [380, 80], [564, 170], [470, 247]],  # ROI ID: ROI_1
    "ROI_2": [[100, 200], [200, 180], [300, 250], [150, 300]],  # ROI ID: ROI_2
    # Add more ROIs here
}

# Employee ID mapping
EMPLOYEE_ID_MAPPING = {
    "vishnu": "1110148",
    # Add more names and their corresponding IDs here
}


def is_face_in_roi(bbox_center, roi_points):
    """
    Check if the center of the bounding box is inside the ROI.
    :param bbox_center: Center of the bounding box (x, y)
    :param roi_points: List of points defining the ROI polygon
    :return: True if the point is inside the ROI, False otherwise
    """
    bbox_center = (int(bbox_center[0]), int(bbox_center[1]))
    return cv2.pointPolygonTest(np.array(roi_points, dtype=np.int32), bbox_center, False) >= 0


def load_known_faces(image_paths, names):
    known_face_embeddings = []
    known_names = []

    for image_path, name in zip(image_paths, names):
        image = cv2.imread(image_path)
        faces = app.get(image)
        for face in faces:
            known_face_embeddings.append(face.normed_embedding)
            known_names.append(name)

    return known_face_embeddings, known_names


def log_entry(employee_id):
    """
    Log entry time for the recognized person.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    data = {
        "employee_id": employee_id,
        "status": 1,  # 1 for entry
        "time_stamp": timestamp
    }
    try:
        # Insert or update document in MongoDB
        attendance_collection.update_one(
            {"employee_id": employee_id},
            {"$set": data},
            upsert=True
        )
        print(f"Entry logged for employee ID {employee_id} at {timestamp}")
    except Exception as e:
        print(f"Failed to log entry for employee ID {employee_id}: {e}")


def log_exit(employee_id):
    """
    Log exit time for the recognized person.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    data = {
        "employee_id": employee_id,
        "status": 0,  # 0 for exit
        "time_stamp": timestamp
    }
    try:
        # Insert or update document in MongoDB
        attendance_collection.update_one(
            {"employee_id": employee_id},
            {"$set": data},
            upsert=True
        )
        print(f"Exit logged for employee ID {employee_id} at {timestamp}")
    except Exception as e:
        print(f"Failed to log exit for employee ID {employee_id}: {e}")


def recognize_faces(frame, known_face_embeddings, known_names, rois):
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
            employee_id = EMPLOYEE_ID_MAPPING.get(name, "Unknown")

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
                        log_entry(recognized_faces[name]["employee_id"])

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
                log_exit(recognized_faces[person]["employee_id"])
            del recognized_faces[person]

    return frame


def process_video(video_capture, known_face_embeddings, known_names, rois):
    frame_skip = 10  # Process every 10th frame to reduce load
    frame_count = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (704, 576))  # Adjust based on your camera's resolution

            future = executor.submit(recognize_faces, frame, known_face_embeddings, known_names, rois)
            processed_frame = future.result()

            cv2.imshow('Video', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()


# Specify the paths to the persons' image files
person_image_paths = [
    "/home/force-4/Tracking_face/pics/j2.png",
    "/home/force-4/Tracking_face/pics/vz2.png",
]

person_names = [
    "vishnu",
    "vishnu",
]

# Load the known face embeddings and names
known_face_embeddings, known_names = load_known_faces(person_image_paths, person_names)

# Open the RTSP stream
video_capture = cv2.VideoCapture("rtsp://admin:Admin$1234@192.168.1.104:554/cam/realmonitor?channel=1&subtype=0")

# Run face recognition on the live video stream, within the specified ROIs
process_video(video_capture, known_face_embeddings, known_names, ROIS)
