import cv2
import numpy as np
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor
import onnxruntime

# Initialize FaceAnalysis with a model and detection size
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], model='arcface_r50_v1')
app.prepare(ctx_id=0, det_size=(640, 640))  # Larger detection size for better accuracy

# Define ROI (Region of Interest) points
ROI_POINTS = [[281, 109], [380, 80], [564, 170], [470, 247]]  # Your defined ROI area

def is_face_in_roi(bbox_center, roi_points):
    """
    Check if the center of the bounding box is inside the ROI.
    :param bbox_center: Center of the bounding box (x, y)
    :param roi_points: List of points defining the ROI polygon
    :return: True if the point is inside the ROI, False otherwise
    """
    # Ensure bbox_center is a tuple of (x, y)
    bbox_center = (int(bbox_center[0]), int(bbox_center[1]))

    # Use pointPolygonTest to check if the face's center is within the ROI
    return cv2.pointPolygonTest(np.array(roi_points, dtype=np.int32), bbox_center, False) >= 0

def draw_transparent_roi(frame, roi_points):
    """
    Add a transparent overlay inside the ROI region.
    :param frame: The video frame
    :param roi_points: List of points defining the ROI polygon
    :return: The frame with transparent ROI drawn
    """
    # Create an overlay that is a copy of the original frame
    overlay = frame.copy()

    # Fill the ROI with a transparent color (e.g., light gray with 50% transparency)
    roi_points = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(overlay, [roi_points], color=(192, 192, 192))  # Light gray

    # Add the transparent overlay onto the original frame (0.5 is the transparency factor)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    return frame

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

def recognize_faces(frame, known_face_embeddings, known_names, roi_points):
    distance_threshold = 0.4  # Tighter threshold for better accuracy

    # Perform face detection on the frame
    faces = app.get(frame)
    face_count = len(faces)  # Count the number of detected faces

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
        bbox_center = ((left + right) // 2, (top + bottom) // 2)  # Center of the bounding box

        # Check if the face is inside the ROI
        if is_face_in_roi(bbox_center, roi_points):
            # Draw bounding box and name if inside ROI
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green box for inside ROI
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left, bottom + 15), font, 0.5, (255, 255, 255), 1)
        else:
            # Draw bounding box in yellow if the face is outside the ROI
            if name == "Unknown":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Red box for unknown faces
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)  # Yellow box for outside ROI

    # Display the face count on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

def process_video(video_capture, known_face_embeddings, known_names, roi_points):
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

            # Resize frame to maintain high accuracy while balancing performance
            frame = cv2.resize(frame, (704, 576))  # Adjust as per your camera's capabilities

            # Draw the transparent ROI on the frame
            frame = draw_transparent_roi(frame, roi_points)

            # Submit the frame for recognition processing
            future = executor.submit(recognize_faces, frame, known_face_embeddings, known_names, roi_points)

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
]

person_names = [
    "vishnu",
    "vishnu",
]

# Load the known face embeddings and names
known_face_embeddings, known_names = load_known_faces(person_image_paths, person_names)

# Open the RTSP stream
video_capture = cv2.VideoCapture("rtsp://admin:Admin$1234@192.168.1.104:554/cam/realmonitor?channel=1&subtype=0")

# Run face recognition on the live video stream, within the specified ROI
process_video(video_capture, known_face_embeddings, known_names, ROI_POINTS)
