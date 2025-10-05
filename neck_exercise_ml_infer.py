import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import joblib

# ==============================
# Constants
# ==============================
MP_FACE_MESH = mp.solutions.face_mesh
POINTS_IDX = [
    1,
    152,
    263,
    33,
    287,
    57,
    9,
]  # Nose, Chin, R Eye, L Eye, R Mouth, L Mouth, Forehead
POINTS_NAME = [
    "nose",
    "chin",
    "right_eye",
    "left_eye",
    "mouth_right",
    "mouth_left",
    "forehead",
]

PITCH_UP_THRESH = -25
PITCH_DOWN_THRESH = 25
YAW_RIGHT_THRESH = 45
YAW_LEFT_THRESH = -45

CENTER_PITCH_THRESH = 10
CENTER_YAW_THRESH = 10

HOLD_REQUIRED = 30
MIN_TIME_BETWEEN_REPS = 0.8


# ==============================
# Helper Functions
# ==============================
def smooth_angle(prev, new, alpha=0.8):
    """Smooth out sudden changes in angles."""
    return prev * alpha + new * (1 - alpha)


def extract_and_normalize_landmarks(landmarks, width, height):
    """Extract landmarks and normalize them for ML model input."""
    landmark_dict = {}

    # Extract raw landmark coordinates
    for name, idx in zip(POINTS_NAME, POINTS_IDX):
        landmark = landmarks[idx]
        landmark_dict[f"{name}_x"] = landmark.x * width
        landmark_dict[f"{name}_y"] = landmark.y * height

    df_landmarks = pd.DataFrame([landmark_dict])

    # Center around nose
    nose_x = df_landmarks["nose_x"].values[0]
    nose_y = df_landmarks["nose_y"].values[0]

    for col in df_landmarks.columns:
        if col.endswith("_x") and col != "nose_x":
            df_landmarks[col] -= nose_x
        elif col.endswith("_y") and col != "nose_y":
            df_landmarks[col] -= nose_y

    df_landmarks["nose_x"] = 0.0
    df_landmarks["nose_y"] = 0.0

    # Scale normalization
    scale_x = abs(
        df_landmarks["right_eye_x"].values[0] - df_landmarks["left_eye_x"].values[0]
    )
    scale_y = abs(
        df_landmarks["forehead_y"].values[0] - df_landmarks["nose_y"].values[0]
    )

    for col in df_landmarks.columns:
        if col.endswith("_x"):
            df_landmarks[col] /= scale_x
        elif col.endswith("_y"):
            df_landmarks[col] /= scale_y

    return df_landmarks


def get_current_direction(pitch, yaw):
    """Determine head direction based on pitch and yaw angles."""
    if (-CENTER_PITCH_THRESH <= pitch <= CENTER_PITCH_THRESH) and (
        -CENTER_YAW_THRESH <= yaw <= CENTER_YAW_THRESH
    ):
        return "Center"
    elif pitch < PITCH_UP_THRESH and (-CENTER_YAW_THRESH <= yaw <= CENTER_YAW_THRESH):
        return "Up"
    elif pitch > PITCH_DOWN_THRESH and (-CENTER_YAW_THRESH <= yaw <= CENTER_YAW_THRESH):
        return "Down"
    elif yaw < YAW_LEFT_THRESH and (
        -CENTER_PITCH_THRESH <= pitch <= CENTER_PITCH_THRESH
    ):
        return "Right"
    elif yaw > YAW_RIGHT_THRESH and (
        -CENTER_PITCH_THRESH <= pitch <= CENTER_PITCH_THRESH
    ):
        return "Left"
    else:
        return "Not Detected"


def draw_developer_overlay(
    frame, pitch, yaw, cur_dir, hold_count, landmarks, width, height
):
    """Draw angles, direction, and landmarks on frame."""
    # Pitch & Yaw
    cv2.putText(
        frame,
        f"Pitch: {pitch:.2f}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Yaw: {yaw:.2f}",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    # Direction
    color = (0, 255, 255) if cur_dir != "Not Detected" else (0, 0, 255)
    cv2.putText(
        frame,
        f"Direction: {cur_dir}  Hold: {hold_count}",
        (50, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        frame,
        "Developer Mode ON (ML)",
        (50, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # 2D Landmarks
    for name, idx in zip(POINTS_NAME, POINTS_IDX):
        x = int(landmarks[idx].x * width)
        y = int(landmarks[idx].y * height)
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"{name}",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )


def draw_neck_reps(frame, neck_reps):
    """Display rep counts on frame."""
    y_start = 200
    for i, (direction, count) in enumerate(neck_reps.items()):
        cv2.putText(
            frame,
            f"{direction}: {count}",
            (50, y_start + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
        )


# ==============================
# Main
# ==============================
def main():
    # Load trained ML model and scaler
    try:
        model = joblib.load("trained_model.pkl")
        scaler = joblib.load("scaler.pkl")
        print("ML model and scaler loaded successfully!")
    except FileNotFoundError:
        print("Error: trained_model.pkl or scaler.pkl not found!")
        print("Please train the model first using the training code.")
        return

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot access camera")
        return

    height, width = frame.shape[:2]

    face_mesh = MP_FACE_MESH.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )

    prev_pitch, prev_yaw = 0, 0
    developer_mode = True
    neck_reps = {"Up": 0, "Down": 0, "Right": 0, "Left": 0}
    last_rep_time = 0.0
    sequence = []  # store compressed sequence of directions
    hold_dir = None
    hold_count = 0
    pattern_text = ""

    cv2.namedWindow("ML Head Pose Estimation", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "ML Head Pose Estimation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        cur_dir = "Not Detected"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Extract and normalize landmarks for ML model
            df_landmarks = extract_and_normalize_landmarks(
                face_landmarks, width, height
            )

            # Predict pitch and yaw using ML model
            X_input = scaler.transform(df_landmarks)
            pitch, yaw = model.predict(X_input)[0]

            # Smooth angles
            pitch = smooth_angle(prev_pitch, pitch)
            yaw = smooth_angle(prev_yaw, yaw)
            prev_pitch, prev_yaw = pitch, yaw

            # Determine current direction
            cur_dir = get_current_direction(pitch, yaw)

            # --- Hold logic for directional movements only ---
            if cur_dir in ("Up", "Down", "Left", "Right"):
                if hold_dir is None or hold_dir != cur_dir:
                    hold_dir = cur_dir
                    hold_count = 1
                else:
                    hold_count += 1
            else:
                # When leaving a directional position, check if it was held long enough
                if hold_dir is not None and hold_count >= HOLD_REQUIRED:
                    if not sequence or sequence[-1] != hold_dir:
                        sequence.append(hold_dir)

                # Reset hold tracking
                hold_dir = None
                hold_count = 0

                # Append Center or Not Detected only if different from last element
                if not sequence or sequence[-1] != cur_dir:
                    sequence.append(cur_dir)

            # --- Check for valid pattern: Center -> ND -> Dir -> ND -> Center ---
            if len(sequence) >= 5:
                if (
                    sequence[0] == "Center"
                    and sequence[1] == "Not Detected"
                    and sequence[2] in ("Up", "Down", "Left", "Right")
                    and sequence[3] == "Not Detected"
                    and sequence[4] == "Center"
                ):
                    now = time.time()
                    if (now - last_rep_time) > MIN_TIME_BETWEEN_REPS:
                        neck_reps[sequence[2]] += 1
                        last_rep_time = now
                        pattern_text = f"REP COUNTED: {sequence[2]}"
                        print(pattern_text)
                    # Remove the matched pattern
                    sequence = sequence[5:]
                else:
                    # Invalid pattern - remove first element and try again
                    sequence.pop(0)

            # Prevent sequence from growing too large
            if len(sequence) > 10:
                sequence.pop(0)

            # --- Developer overlay ---
            if developer_mode:
                draw_developer_overlay(
                    frame,
                    pitch,
                    yaw,
                    cur_dir,
                    hold_count,
                    face_landmarks,
                    width,
                    height,
                )

        # Draw rep counts
        draw_neck_reps(frame, neck_reps)

        cv2.imshow("ML Head Pose Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("d"):
            developer_mode = not developer_mode

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
