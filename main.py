import cv2
import numpy as np
import supervision as sv
from ultralytics import SAM
import torch
from tqdm import tqdm


def click_event(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        print(f"Point selected: {point}")


def detect_gpu():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    else:
        print("No GPU detected.")
        exit()

device = detect_gpu()

# Load a model
model = SAM("sam2_b.pt")

video_path = "XVR_ch1_main_20210910141900_20210910142500_chunk_3_of_6.mp4"
target_path = video_path.replace(".mp4", "_segmented.mp4")
frame_generator = sv.get_video_frames_generator(source_path=video_path)
video_info = sv.VideoInfo.from_video_path(video_path=video_path)

mask_annotator = sv.PolygonAnnotator()
box_annotator = sv.BoxAnnotator()
labeler = sv.LabelAnnotator()

# Initialize global variables to store the point coordinates
point = None


cv2.namedWindow('Predictions')
cv2.setMouseCallback('Predictions', click_event)

# Open the video sink
with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        if point is not None:
            # Use the selected point for processing
            results = next(model(frame, 
                            stream=True,
                            points=[point], 
                            labels=[1], 
                            device = device))[0]
            
            masks = results.masks.data.cpu().numpy()
            xyxy = sv.mask_to_xyxy(masks)
            cls = np.array([0] * len(xyxy))
            detections = sv.Detections(xyxy, masks, class_id=cls)

            # Annotate the frame with boxes, masks, and labels
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = mask_annotator.annotate(scene=frame, detections=detections)

        # Display frame and wait for a click event
        cv2.imshow('Predictions', frame)
        key = cv2.waitKey(1)

        # Break the loop if the 'q' key is pressed
        if key & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()