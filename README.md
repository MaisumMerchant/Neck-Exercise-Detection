# Neck Exercise Detection Application

A real-time application for tracking and counting neck exercises using computer vision and head pose estimation. The application uses facial landmarks to detect head movements and counts repetitions of neck exercises in both vertical (up-down) and horizontal (left-right) directions.

## Features

-   Real-time head pose estimation using MediaPipe Face Mesh
-   Tracks and counts neck exercises:
    -   Up-Down movements
    -   Left-Right movements
-   Interactive GUI with Kivy
-   Developer mode for detailed visualization
-   Adjustable threshold controls for movement detection
-   Fullscreen support
-   Live visual feedback with threshold regions

## Requirements

-   Python
-   Webcam

Dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd neck-exercise-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python neck_exercise.py
```

### Controls

-   **Toggle Dev Mode**: Enables/disables visualization of facial landmarks, angles, and threshold regions
-   **Reset Counters**: Resets both vertical and horizontal exercise counters
-   **Toggle Fullscreen**: Switch between fullscreen and windowed mode

### Threshold Controls

The application provides adjustable sliders for fine-tuning detection parameters:

-   **Pitch Up/Down**: Angle thresholds for up/down movements
-   **Yaw Left/Right**: Angle thresholds for left/right movements
-   **Center Pitch/Yaw**: Defines the "center" position tolerance
-   **Hold Frames**: Required frames to hold a position
-   **Min Time**: Minimum time between repetitions
-   **Smoothing**: Movement smoothing factor

## How It Works

1. The application captures video from your webcam
2. MediaPipe Face Mesh detects facial landmarks
3. Head pose (pitch and yaw angles) is estimated using these landmarks
4. Movements are classified based on the angles and threshold settings
5. Exercise repetitions are counted when a complete movement pattern is detected

## Technical Details

-   Uses MediaPipe Face Mesh for facial landmark detection
-   Implements PnP (Perspective-n-Point) algorithm for head pose estimation
-   Kivy-based GUI for cross-platform compatibility
-   Real-time processing with OpenCV

## Project Structure

-   `neck_exercise.py`: Main application file containing:
    -   Head pose estimation logic
    -   Movement pattern recognition
    -   GUI implementation
    -   Visualization utilities

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is open source and available under the MIT License.
