import cv2
import numpy as np
from typing import Tuple, List, Optional
from src.core.uav_detector import UAVDetector

# ========== CONFIGURATION ==========
INPUT_VIDEO_PATH = "../ignore/video_test/swarm-01.mp4"  # Path to input video
OUTPUT_VIDEO_PATH = "../ignore/video_out/swarm-01-.mp4"  # Path to save output video
MODEL_PATH = "../models/yolov11n-UAV-finetune.pt"
CONF_THRESHOLD = 0.50
DEVICE = "mps"  # "cpu", "cuda", or "mps"
# ===================================


class MultiUAVDetector:
    """
    Detection system for multiple UAVs (no tracking, just detection on each frame).
    Returns ALL detected drones, not just one.
    """

    def __init__(
        self,
        model_path: str = "../models/yolov11n-UAV-finetune.pt",
        conf_threshold: float = 0.30,
        device: str = "cpu",
    ):
        """
        Initialize the multi-UAV detector.

        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detection
            device: Device to run model on ("cpu", "cuda", or "mps")
        """
        self.detector = UAVDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect ALL UAVs in the frame.

        Args:
            frame: Input frame from camera

        Returns:
            Tuple of (annotated_frame, bboxes_list) where:
                - annotated_frame: Frame with ALL detections visualized
                - bboxes_list: List of bounding boxes [[x1, y1, x2, y2], ...], empty if no detections
        """
        # Use the detector's slicer directly to get ALL detections
        detections = self.detector.slicer(frame)

        if len(detections) == 0:
            return frame, []

        # Get all bounding boxes
        bboxes_list = [bbox for bbox in detections.xyxy]

        # Annotate frame with ALL detections
        annotated_frame = frame.copy()

        # Create labels for all detections
        labels = [
            f"UAV {conf:.2f}"
            for conf in detections.confidence
        ]

        # Annotate with supervision
        annotated_frame = self.detector.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = self.detector.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        return annotated_frame, bboxes_list


def process_video():
    """
    Process video file with multi-UAV detection and save annotated output.
    """
    # Initialize multi-UAV detector
    print("Loading model...")
    detector = MultiUAVDetector(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        device=DEVICE,
    )
    print("Model loaded successfully!")

    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {INPUT_VIDEO_PATH}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {INPUT_VIDEO_PATH}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Cannot create output video: {OUTPUT_VIDEO_PATH}")
        cap.release()
        return

    print(f"Processing video...")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame with multi-UAV detector
            annotated_frame, bboxes_list = detector.process_frame(frame)

            # Write annotated frame to output video
            out.write(annotated_frame)

            # Print progress with number of detections
            if frame_count % 30 == 0:  # Print every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - {len(bboxes_list)} UAVs detected")

    finally:
        cap.release()
        out.release()
        print(f"\nProcessing complete!")
        print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    process_video()
