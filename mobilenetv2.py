import numpy as np # loads in numpy library
import cv2 # loads in OpenCV (open-source library for computer vision)
import time # loads in python's built-in time module (used for FPS tracking)
import os # loads in python's built-in operating system module 
import urllib.request # loads in python's built-in URL request module (used to download the pre-trained model files)

try: 
    import mediapipe as mp 
    from mediapipe.tasks import python as mp_python 
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode 
except ImportError:
    print("mediapipe not found - install with: pip install mediapipe")
    exit(1)

print(f"OpenCV version: {cv2.__version__}")

# Configuration
class Config:
    CONFIDENCE_THRESHOLD = 0.6
    IOU_THRESHOLD = 0.4
    CAMERA_INDEX = 0
    DISPLAY_SCALE = 4
    # Pre-trained face model files
    MODEL_CONFIG = "deploy.prototxt"
    MODEL_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"
    MODEL_CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    MODEL_WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    # MediaPipe hand landmarker model
    HAND_MODEL_PATH = "hand_landmarker.task"
    HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

config = Config()

# Hand landmark connections (21 keypoints)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),       # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),     # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),     # Pinky
    (5, 9), (9, 13), (13, 17),                 # Palm
]

# Colors
FACE_COLOR  = (0, 255, 0)      # Green for face
HAND_COLOR  = (0, 200, 255)    # Orange for hand skeleton
JOINT_COLOR = (255, 255, 255)  # White for joints


def download_model_files():
    # Download pre-trained model files if not present
    if not os.path.exists(config.MODEL_CONFIG):
        print("Downloading face model config...")
        urllib.request.urlretrieve(config.MODEL_CONFIG_URL, config.MODEL_CONFIG)
        print("  Done.")

    if not os.path.exists(config.MODEL_WEIGHTS):
        print("Downloading face model weights (~10MB)...")
        urllib.request.urlretrieve(config.MODEL_WEIGHTS_URL, config.MODEL_WEIGHTS)
        print("  Done.")

    if not os.path.exists(config.HAND_MODEL_PATH):
        print("Downloading hand landmarker model (~9MB)...")
        urllib.request.urlretrieve(config.HAND_MODEL_URL, config.HAND_MODEL_PATH)
        print("  Done.")


def non_max_suppression(boxes, scores, iou_threshold):
    """Non-maximum suppression to remove overlapping boxes"""
    if len(boxes) == 0:
        return []

    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        ious = compute_iou(current_box, other_boxes)
        indices = indices[1:][ious < iou_threshold]

    return keep


def compute_iou(box, boxes):
    """Compute IoU between one box and multiple boxes"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    iou = intersection / (union + 1e-6)

    return iou


def decode_predictions(detections, conf_threshold=0.6, iou_threshold=0.4):
    """Decode face model predictions to bounding boxes"""
    boxes = []
    scores = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(0, detections[0, 0, i, 3])
            y1 = max(0, detections[0, 0, i, 4])
            x2 = min(1, detections[0, 0, i, 5])
            y2 = min(1, detections[0, 0, i, 6])
            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))

    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = non_max_suppression(boxes, scores, iou_threshold)

    return boxes[indices], scores[indices]


class FaceDetectorDemo:
    """Face + Hand detector with webcam support"""

    def __init__(self, model_path=None):
        print("\n[1/4] Initializing face detector...")
        download_model_files()
        self.face_model = cv2.dnn.readNetFromCaffe(config.MODEL_CONFIG, config.MODEL_WEIGHTS)
        print("Face detection model loaded successfully")

        # Initialize MediaPipe hand landmarker (new API)
        print("\n[2/4] Initializing hand detector...")
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_model = HandLandmarker.create_from_options(options)
        self.frame_timestamp = 0
        print("Hand detection model loaded successfully")

        # Initialize camera
        print("\n[3/4] Initializing camera...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Camera initialized successfully")
        print("\n[4/4] Setup complete!\n")

        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def preprocess_frame(self, frame):
        """Preprocess frame for face model input"""
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        return blob

    def detect_faces(self, frame):
        """Detect faces in frame"""
        blob = self.preprocess_frame(frame)
        self.face_model.setInput(blob)
        detections = self.face_model.forward()
        boxes, scores = decode_predictions(
            detections,
            config.CONFIDENCE_THRESHOLD,
            config.IOU_THRESHOLD
        )
        return boxes, scores

    def detect_hands(self, frame):
        """Detect hands using new MediaPipe Tasks API"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_timestamp += 1
        result = self.hand_model.detect_for_video(mp_image, self.frame_timestamp)
        return result.hand_landmarks if result.hand_landmarks else []

    def draw_detections(self, frame, boxes, scores):
        """Draw face bounding boxes on frame"""
        h, w = frame.shape[:2]

        for box, score in zip(boxes, scores):
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), FACE_COLOR, 2)

            label = f"Face: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), FACE_COLOR, -1)
            cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)

        return frame

    def draw_hand_skeleton(self, frame, hand_landmarks_list):
        """Draw hand skeletal structure with joints and bones"""
        h, w = frame.shape[:2]

        for hand_landmarks in hand_landmarks_list:
            # Get all landmark pixel positions
            points = {}
            for idx, lm in enumerate(hand_landmarks):
                px = int(lm.x * w)
                py = int(lm.y * h)
                points[idx] = (px, py)

            # Draw bones (connections between joints)
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx in points and end_idx in points:
                    cv2.line(frame, points[start_idx], points[end_idx], HAND_COLOR, 2)

            # Draw joints (circles at each landmark)
            for idx, (px, py) in points.items():
                radius = 5 if idx in [4, 8, 12, 16, 20] else 3
                cv2.circle(frame, (px, py), radius, JOINT_COLOR, -1)
                cv2.circle(frame, (px, py), radius, HAND_COLOR, 1)

            # Label at wrist
            wrist = points[0]
            cv2.putText(frame, "Hand", (wrist[0] - 20, wrist[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, HAND_COLOR, 2)

        return frame

    def draw_info(self, frame, num_faces, num_hands):
        """Draw info overlay"""
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {num_faces}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, FACE_COLOR, 2)
        cv2.putText(frame, f"Hands: {num_hands}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, HAND_COLOR, 2)
        cv2.putText(frame, "Press 'q' to quit | 's' to save frame",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

    def run(self):
        """Run face + hand detection on webcam"""
        print("=" * 60)
        print("FACE + HAND DETECTION DEMO")
        print("=" * 60)
        print("\nWhat's detected:")
        print("  Green box       = Face")
        print("  Orange skeleton = Hand")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("=" * 60)
        print("\nStarting detection...\n")

        saved_frame_count = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                face_boxes, face_scores = self.detect_faces(frame)
                hand_landmarks = self.detect_hands(frame)

                frame = self.draw_detections(frame, face_boxes, face_scores)
                frame = self.draw_hand_skeleton(frame, hand_landmarks)
                frame = self.draw_info(frame, len(face_boxes), len(hand_landmarks))
                self.update_fps()

                cv2.imshow('TinyML Face + Hand Detection', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    filename = f"detected_face_{saved_frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame: {filename}")
                    saved_frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            print("\nCleaning up...")
            self.cap.release()
            self.hand_model.close()
            cv2.destroyAllWindows()

            print("\n" + "=" * 60)
            print(f"Session Summary:")
            print(f"  Frames processed: {self.frame_count}")
            print(f"  Average FPS: {self.fps:.1f}")
            print(f"  Frames saved: {saved_frame_count}")
            print("=" * 60)


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("TinyML Face + Hand Detection - Laptop Demo")
    print("=" * 60)

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("\nGPU available, enabling CUDA")
    else:
        print("\nNo GPU found, using CPU")

    try:
        detector = FaceDetectorDemo()
        detector.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()