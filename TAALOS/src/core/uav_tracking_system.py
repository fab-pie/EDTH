import numpy as np
from typing import Optional, Tuple
from src.core.uav_detector import UAVDetector
from src.core.uav_tracker import UAVTracker


class UAVTrackingSystem:
    """
    Main UAV tracking system that combines detection and tracking.
    Implements a state machine that switches between detection and tracking modes.
    """

    def __init__(
        self,
        model_path: str = "../models/yolov11n-UAV-finetune.pt",
        conf_threshold: float = 0.30,
        device: str = "cpu",
    ):
        """
        Initialize the UAV tracking system.

        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detection
            device: Device to run model on ("cpu", "cuda", or "mps")
        """

        # Initialize detector and tracker
        self.detector = UAVDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device
        )

        self.tracker = UAVTracker(
            border_margin_px=10,
            min_bbox_area=50 * 50,
            max_bbox_area_ratio=0.1,
        )

    @property
    def is_tracking(self) -> bool:
        """Returns True if currently tracking a UAV."""
        return self.tracker.is_tracking

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a single frame: detect or track UAV.

        Args:
            frame: Input frame from camera

        Returns:
            Tuple of (annotated_frame, bbox_xyxy) where:
                - annotated_frame: Frame with detection/tracking visualization
                - bbox_xyxy: Bounding box [x1, y1, x2, y2] or None if no UAV detected/tracked
        """
        # ---------- MODE DETECTION ----------
        if not self.tracker.is_tracking:
            bbox_xyxy, annotated_frame = self.detector.detect(
                frame,
                annotate=True
            )

            if bbox_xyxy is not None:


                # Start tracking
                success = self.tracker.start(frame, bbox_xyxy)


            return annotated_frame, bbox_xyxy

        # ---------- MODE TRACKING ----------
        bbox_xyxy, annotated_frame = self.tracker.update(
            frame,
            annotate=True
        )

        if bbox_xyxy is None:
            # Drone lost -> return to detection mode
            self.tracker.stop()

        return annotated_frame, bbox_xyxy

    def get_current_bbox(self) -> Optional[np.ndarray]:
        """
        Get current tracked UAV bounding box.

        Returns:
            Bbox in [x1, y1, x2, y2] format, or None if not tracking
        """
        return self.tracker.current_bbox
