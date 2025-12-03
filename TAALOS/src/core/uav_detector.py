import supervision as sv
from ultralytics import YOLO
import numpy as np
from typing import Optional, Tuple


class UAVDetector:
    """
    YOLO-based UAV detector using InferenceSlicer for better detection on high-resolution images.
    Detects UAVs and returns the single drone with highest confidence.
    """

    def __init__(
        self,
        model_path: str = "../models/yolov11n-UAV-finetune.pt",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.3,
        slice_wh: Tuple[int, int] = (640, 640), # or (960, 540)
        overlap_wh: Tuple[int, int] = (0, 0),
        device: str = "cpu"
    ):
        """
        Initialize the UAV detector.

        Args:
            model_path: Path to the YOLO model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            slice_wh: Width and height of each slice for InferenceSlicer
            overlap_wh: Overlap between slices (width, height)
        """
        self.model = YOLO(model_path).to(device=device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Setup supervision annotators
        self.color = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
            "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
        ])
        self.box_annotator = sv.BoxAnnotator(color=self.color, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=self.color,
            text_color=sv.Color.BLACK
        )

        # Setup InferenceSlicer
        self.slicer = sv.InferenceSlicer(
            callback=self._slicer_callback,
            slice_wh=slice_wh,
            iou_threshold=self.iou_threshold,
            overlap_ratio_wh=None,
            overlap_wh=overlap_wh,
        )

    def _slicer_callback(self, image_slice: np.ndarray) -> sv.Detections:
        """
        Callback function called by InferenceSlicer on each tile.

        Args:
            image_slice: Image slice to run inference on

        Returns:
            Detections object from supervision
        """
        result = self.model(image_slice, conf=self.conf_threshold, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def detect(
        self,
        frame: np.ndarray,
        annotate: bool = True,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect UAVs in the given frame and return the best detection.

        Args:
            frame: Input image frame
            annotate: Whether to annotate the frame with detection results

        Returns:
            Tuple of (bbox_xyxy, annotated_frame) where:
                - bbox_xyxy: numpy array [x1, y1, x2, y2] of the best detection, or None
                - annotated_frame: frame with annotations (or original if annotate=False)
        """
        detections = self.slicer(frame)


        if len(detections) == 0:
            return None, frame

        # Select the detection with highest confidence
        best_idx = int(np.argmax(detections.confidence))
        bbox_xyxy = detections.xyxy[best_idx]  # shape (4,)
        conf = float(detections.confidence[best_idx])

        # Handle class_id safely
        if detections.class_id is not None:
            class_id_val = int(detections.class_id[best_idx])
        else:
            class_id_val = 0

        if not annotate:
            return np.array(bbox_xyxy, dtype=float), frame

        # Create single detection for annotation
        single_det = sv.Detections(
            xyxy=np.array(bbox_xyxy, dtype=float).reshape(-1, 4),
            confidence=np.array([conf], dtype=float),
            class_id=np.array([class_id_val], dtype=int),
        )

        labels = [f"UAV {conf:.2f}"]

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=single_det
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=single_det,
            labels=labels
        )

        return np.array(bbox_xyxy, dtype=float), annotated_frame
