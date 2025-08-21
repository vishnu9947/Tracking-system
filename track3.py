import cv2
import numpy as np
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor
import onnxruntime

# Initialize FaceAnalysis with a model and detection size
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], model='arcface_r50_v1')
app.prepare(ctx_id=0, det_size=(640, 640))  # Larger detection size for better accuracy

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

def recognize_faces(frame, known_face_embeddings, known_names):
    distance_threshold = 0.4  # Tighter threshold for better accuracy

    # Perform face detection on the frame
    faces = app.get(frame)

    for face in faces:
        # Compute cosine similarity between detected faces and known faces
        distances = np.dot(known_face_embeddings, face.normed_embedding.T)
        max_similarity_index = np.argmax(distances)
        name = "Unknown"

        if distances[max_similarity_index] > distance_threshold:
            name = known_names[max_similarity_index]

        # Scale bounding box to original frame size
        bbox = face.bbox.astype(int)
        left, top, right, bottom = bbox
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, bottom + 15), font, 0.5, (255, 255, 255), 1)

    return frame

def process_video(video_capture, known_face_embeddings, known_names):
    frame_skip = 10  # Process every 5th frame to reduce load
    frame_count = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Resize frame to maintain high accuracy while balancing performance
            frame = cv2.resize(frame, (704, 576))  # Adjust as per your camera's capabilities

            # Submit the frame for recognition processing
            future = executor.submit(recognize_faces, frame, known_face_embeddings, known_names)

            # Show the processed frame
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
    "/home/force-4/Tracking_face/pics/j1.png",
    "/home/force-4/Tracking_face/pics/g1.png",
    "/home/force-4/Tracking_face/pics/g2.png",
    "/home/force-4/Tracking_face/pics/g3.png",
    "/home/force-4/Tracking_face/pics/r1.png",
    "/home/force-4/Tracking_face/pics/r2.png",
    "/home/force-4/Tracking_face/pics/r3.png",
    "/home/force-4/Tracking_face/pics/k1.png",
    "/home/force-4/Tracking_face/pics/k2.png",
    "/home/force-4/Tracking_face/pics/k3.png",
    "/home/force-4/Tracking_face/pics/n1.png",
    "/home/force-4/Tracking_face/pics/n2.png",
    "/home/force-4/Tracking_face/pics/a1.png",
    "/home/force-4/Tracking_face/pics/a2.png",
    "/home/force-4/Tracking_face/pics/a3.png",
]

person_names = [
    "vishnu",
    "vishnu",
    "vishnu",
    "Gwori",
    "Gwori",
    "Gwori",
    "rahul",
    "rahul",
    "rahul",
    "kiran",
    "kiran",
    "kiran",
    "Nirmal",
    "Nirmal",
    "Arjun",
    "Arjun",
    "Arjun",
]

# Load the known face embeddings and names
known_face_embeddings, known_names = load_known_faces(person_image_paths, person_names)

# Open the RTSP stream
video_capture = cv2.VideoCapture("rtsp://admin:Admin$1234@192.168.1.104:554/cam/realmonitor?channel=1&subtype=0")

# Run face recognition on the live video stream
process_video(video_capture, known_face_embeddings, known_names)
