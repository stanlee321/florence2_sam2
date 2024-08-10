# Video Segmentation with SAM and Supervision

This project performs video segmentation using the SAM model and Supervision library. It processes video frames, detects objects, and annotates them with bounding boxes and masks.

## Prerequisites

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Supervision](https://github.com/roboflow/supervision)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [tqdm](https://github.com/tqdm/tqdm)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/video-segmentation.git
    cd video-segmentation
    ```

2. Install the required packages:
    ```sh
    pip install torch opencv-python numpy supervision ultralytics tqdm
    ```

## Usage

1. Ensure you have a video file in the same directory as `main.py`. The video file should be in `.mp4` format.

2. Place your SAM model file (`.pt` format) in the same directory as `main.py`.

3. Run the script:
    ```sh
    python main.py
    ```

## Code Overview

- `main.py`: The main script that loads the video, processes each frame, and performs segmentation using the SAM model.

### Key Functions

- `detect_gpu()`: Checks if a GPU is available and returns the device name.
- `click_event(event, x, y, flags, param)`: Handles mouse click events to select points on the video frames.

### Main Workflow

1. Detects if a GPU is available using `detect_gpu()`.
2. Loads the SAM model.
3. Reads the video file and initializes annotators.
4. Processes each frame of the video:
    - If a point is selected, it uses the SAM model to generate segmentation masks.
    - Annotates the frame with bounding boxes and masks.
    - Displays the annotated frame.
5. Exits the loop if the 'q' key is pressed.

## .gitignore

The `.gitignore` file is configured to ignore video files (`*.mp4`) and model files (`*.pt`).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.