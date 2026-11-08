import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# --- Модель ---
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    url = ('https://storage.googleapis.com/mediapipe-models/'
           'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# --- Индексы точек глаз ---
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def bbox_from_points(points, w, h, pad=2):
    xs = [p.x * w for p in points]
    ys = [p.y * h for p in points]
    x1, y1 = int(min(xs)) - pad, int(min(ys)) - pad
    x2, y2 = int(max(xs)) + pad, int(max(ys)) + pad
    return max(x1, 0), max(y1, 0), x2, y2

cap = cv2.VideoCapture(0)
ts = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )

    result = landmarker.detect_for_video(mp_img, ts)
    ts += 33

    for face_lms in result.face_landmarks:
        # --- Рамка лица ---
        x1, y1, x2, y2 = bbox_from_points(face_lms, w, h, pad=0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Рамки глаз ---
        for idx_list in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            pts = [face_lms[i] for i in idx_list]
            ex1, ey1, ex2, ey2 = bbox_from_points(pts, w, h, pad=2)
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 255, 255), 1)

    cv2.imshow('Face + Eyes', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()