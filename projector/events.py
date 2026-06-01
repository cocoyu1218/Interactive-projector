import os
import time
import urllib.request

import cv2
import numpy as np

from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode
from mediapipe.tasks.python.core import base_options
from collections import deque

MODEL_PATH = "hand_landmarker.task"
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-tasks/models/hand_landmarker.task",
]

WALL_DEPTH = -0.05
TOUCH_THRESHOLD = 0.02

last_tap_time = 0
TAP_COOLDOWN = 0.30

tap_start_time = None

TAP_HOLD_TIME = 1  # seconds finger must stay still
MOVE_THRESHOLD = 0.03 # movement allowed

last_x = None
last_y = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

OUTPUT_DIR = "outputs"
SAMPLE_IMAGE_PATH = os.path.join(OUTPUT_DIR, "hand_landmarks_sample.png")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "hand_landmarks_output.avi")

def ensure_model(path=MODEL_PATH):
    if os.path.exists(path):
        return path
    for url in MODEL_URLS:
        try:
            print(f"Downloading model from {url}...")
            urllib.request.urlretrieve(url, path)
            print("Model downloaded to", path)
            return path
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    raise RuntimeError(
        "Failed to download MediaPipe hand landmarker model.\n"
        "Please download a hand_landmarker.task model manually and place it next to first.py"
    )

def draw_hand_landmarks(image, hand_landmarks):
    global tap_start_time
    global last_tap_time
    global last_x, last_y

    height, width = image.shape[:2]

    for landmarks in hand_landmarks:

        # Index finger landmarks
        index_tip = landmarks[8]
        index_pip = landmarks[6]

        # Convert to pixel coordinates
        x = int(index_tip.x * width)
        y = int(index_tip.y * height)

        # Draw fingertip
        cv2.circle(image, (x, y), 8, (255, 0, 0), -1)

        cv2.putText(
            image,
            "8",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Finger pointing check
        pointing = index_tip.y < index_pip.y

        current_x = index_tip.x
        current_y = index_tip.y

        # Initialize previous position
        if last_x is None:
            last_x = current_x
            last_y = current_y

        # Movement between frames
        movement = (
            abs(current_x - last_x)
            + abs(current_y - last_y)
        )

        # Show debug info
        cv2.putText(
            image,
            f"Move:{movement:.4f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if pointing:

            cv2.putText(
                image,
                "POINTING",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if movement < MOVE_THRESHOLD:

                if tap_start_time is None:
                    tap_start_time = time.time()

                hold_duration = time.time() - tap_start_time

                cv2.putText(
                    image,
                    f"Hold:{hold_duration:.1f}s",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                if (
                    hold_duration >= TAP_HOLD_TIME
                    and time.time() - last_tap_time > TAP_COOLDOWN
                ):

                    last_tap_time = time.time()

                    print({
                        "type": "tap",
                        "x": round(current_x, 3),
                        "y": round(current_y, 3)
                    })

                    cv2.putText(
                        image,
                        "TAP!",
                        (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        4
                    )

                    tap_start_time = None

            else:
                tap_start_time = None

        else:
            tap_start_time = None

        # Save position for next frame
        last_x = current_x
        last_y = current_y

def main():
    model_file = ensure_model()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_opts = base_options.BaseOptions(model_asset_path=model_file)
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with hand_landmarker.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (frame_width, frame_height))
        if not writer.isOpened():
            print("Warning: Video writer could not be opened. Video will not be saved.")

        start_ms = int(time.time() * 1000)
        frame_idx = 0
        saved_sample = False
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
                timestamp_ms = start_ms + int(frame_idx * (1000 / 30))
                result = landmarker.detect_for_video(mp_img, timestamp_ms)

                if result.hand_landmarks:
                    draw_hand_landmarks(frame, result.hand_landmarks)
                    print(f"Timestamp: {timestamp_ms} ms — Hands: {len(result.hand_landmarks)}")
                    if not saved_sample:
                        cv2.imwrite(SAMPLE_IMAGE_PATH, frame)
                        print("Saved sample image to", SAMPLE_IMAGE_PATH)
                        saved_sample = True

                if writer.isOpened():
                    writer.write(frame)

                cv2.imshow('Hand Landmarker', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1
        finally:
            cap.release()
            if writer.isOpened():
                writer.release()
            cv2.destroyAllWindows()

if __name__ == '_main_':
    main()