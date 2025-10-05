import cv2
import mediapipe as mp
import numpy as np
import time

# ==============================
# Constants
# ==============================
MP_FACE_MESH = mp.solutions.face_mesh
POINTS_IDX = [1, 152, 263, 33, 287, 57, 9]  # Nose, Chin, L/R Eye, L/R Mouth, Forehead

PITCH_UP_THRESH = -35
PITCH_DOWN_THRESH = 35
YAW_RIGHT_THRESH = 45
YAW_LEFT_THRESH = -45

CENTER_PITCH_THRESH = 15
CENTER_YAW_THRESH = 10

HOLD_REQUIRED = 30
MIN_TIME_BETWEEN_REPS = 0.8

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -400.0, -75.0),  # Chin
        (-225.0, 170.0, -125.0),  # Left eye left corner
        (225.0, 170.0, -125.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
        (0.0, 350.0, -75.0),  # Forehead
    ],
    dtype=np.float64,
)


# ==============================
# Helper Functions
# ==============================
def smooth_angle(prev, new, alpha=0.8):
    """Smooth out sudden changes in angles."""
    return prev * alpha + new * (1 - alpha)


def get_2d_landmarks(landmarks, img_w, img_h):
    """Extract 2D image points from selected landmarks."""
    points = [
        (landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in POINTS_IDX
    ]
    return np.array(points, dtype=np.float64)


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
    frame,
    pitch,
    yaw,
    cur_dir,
    hold_count,
    image_points,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
):
    """Draw angles, direction, landmarks, and projected 3D model points on frame."""
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
        "Developer Mode ON",
        (50, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # 2D Landmarks
    for (x, y), idx in zip(image_points, POINTS_IDX):
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"{idx}",
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    # Projected 3D Model Points
    projected_points, _ = cv2.projectPoints(
        MODEL_POINTS, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = np.squeeze(projected_points)
    for i, (x, y) in enumerate(projected_points):
        cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv2.putText(
            frame,
            f"M{i}",
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


def draw_neck_reps(frame, neck_reps):
    """Display rep counts on frame."""
    y_start = 200
    for i, (dir, count) in enumerate(neck_reps.items()):
        cv2.putText(
            frame,
            f"{dir}: {count}",
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
    cap = cv2.VideoCapture(0)
    width, height = int(cap.get(3)), int(cap.get(4))
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

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

    cv2.namedWindow("Head Pose Estimation", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Head Pose Estimation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
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
            face_landmarks = results.multi_face_landmarks[0]
            image_points = get_2d_landmarks(face_landmarks.landmark, width, height)

            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs
            )
            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                proj_mat = np.hstack((rmat, tvec))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)
                pitch, yaw, _ = euler_angles.flatten()

                pitch = smooth_angle(prev_pitch, pitch)
                yaw = smooth_angle(prev_yaw, yaw)
                prev_pitch, prev_yaw = pitch, yaw

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
                        image_points=image_points,
                        rvec=rvec,
                        tvec=tvec,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                    )

        # Draw rep counts
        draw_neck_reps(frame, neck_reps)

        cv2.imshow("Head Pose Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("d"):
            developer_mode = not developer_mode

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
