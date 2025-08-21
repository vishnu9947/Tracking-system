import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis with a model and detection size
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], model='arcface_r50_v1')
app.prepare(ctx_id=0, det_size=(320, 320))  # Adjust detection size if needed

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

def recognize_faces(video_capture, known_face_embeddings, known_names):
    frame_skip = 5  # Process every 5th frame to reduce load
    frame_count = 0
    distance_threshold = 0.6  # Adjust distance threshold for face recognition

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize frame to reduce processing load
        small_frame = cv2.resize(frame, (640, 480))  # Set to a smaller resolution

        # Perform face detection on the resized frame
        faces = app.get(small_frame)

        for face in faces:
            # Compute distances between detected faces and known faces
            distances = np.linalg.norm(np.array(known_face_embeddings) - face.normed_embedding, axis=1)
            min_distance_index = np.argmin(distances)
            name = "Unknown"

            if distances[min_distance_index] < distance_threshold:
                name = known_names[min_distance_index]

            # Scale bounding box to original frame size
            bbox = [int(coord * (frame.shape[1] / small_frame.shape[1])) for coord in face.bbox.astype(int)]
            left, top, right, bottom = bbox
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left, bottom + 15), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Specify the paths to the persons' image files
person_image_paths = [
    "/home/force-4/Desktop/System_monitor_track_OG/pics/j2.png",
    "/home/force-4/Desktop/System_monitor_track_OG/pics/vz2.png",
    "/home/force-4/Desktop/System_monitor_track_OG/pics/j1.png"
]

person_names = [
    "vishnu",
    "vishnu",
    "vishnu"
]

# Load the known face embeddings and names
known_face_embeddings, known_names = load_known_faces(person_image_paths, person_names)

# Open the RTSP stream
video_capture = cv2.VideoCapture("rtsp://admin:Admin$1234@192.168.1.104:554/cam/realmonitor?channel=1&subtype=0")

# Run face recognition on the live video stream
recognize_faces(video_capture, known_face_embeddings, known_names)
