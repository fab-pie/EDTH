import cv2
import numpy as np
from typing import Optional, Tuple


class UAVTracker:
    """
    OpenCV CSRT-based tracker for maintaining lock on a single UAV.
    Handles tracker initialization, updates, and loss detection.
    """

    def __init__(
        self,
        border_margin_px: int = 10,
        min_bbox_area: int = 50 * 50,
        max_bbox_area_ratio: float = 0.1,
    ):
        """
        Initialize the UAV tracker.

        Args:
            border_margin_px: Margin in pixels from frame border before considering drone lost
            min_bbox_area: Minimum bbox area in pixels (below this = lost)
            max_bbox_area_ratio: Maximum bbox area relative to frame (above this = lost)
        """
        self.border_margin_px = border_margin_px
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area_ratio = max_bbox_area_ratio

        self._tracker: Optional[cv2.legacy.TrackerCSRT] = None
        self._is_tracking = False
        self._current_bbox_xyxy: Optional[np.ndarray] = None

    @property
    def is_tracking(self) -> bool:
        """Returns True if currently tracking a UAV."""
        return self._is_tracking

    @property
    def current_bbox(self) -> Optional[np.ndarray]:
        """Returns current bbox in [x1, y1, x2, y2] format, or None if not tracking."""
        return self._current_bbox_xyxy

    def _create_tracker(self) -> cv2.legacy.TrackerCSRT:
        """
        Create a new CSRT tracker instance.

        Returns:
            OpenCV CSRT tracker
        """
        return cv2.legacy.TrackerCSRT_create()

    def start(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> bool:
        """
        Initialize the tracker with a bounding box.

        Args:
            frame: Current frame
            bbox_xyxy: Bounding box in [x1, y1, x2, y2] format

        Returns:
            True if tracking started successfully, False otherwise
        """
        # Ensure bbox is a 1D array of length 4
        bbox_xyxy = np.array(bbox_xyxy).reshape(-1)

        # Convert to pure Python ints (OpenCV doesn't like NumPy scalars)
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            print("[UAVTracker] Invalid bbox dimensions, cannot start tracking.")
            return False

        self._tracker = self._create_tracker()
        # IMPORTANT: OpenCV expects tuple of pure Python ints
        self._tracker.init(frame, (x1, y1, w, h))

        self._is_tracking = True
        self._current_bbox_xyxy = np.array([x1, y1, x2, y2], dtype=int)

        return True

    def stop(self):
        """Stop tracking and reset state."""
        self._tracker = None
        self._is_tracking = False
        self._current_bbox_xyxy = None

    def update(
        self,
        frame: np.ndarray,
        annotate: bool = True,
        debug: bool = False
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Update tracker with new frame.

        Args:
            frame: Current frame
            annotate: Whether to draw tracking bbox on frame
            debug: Whether to print debug information

        Returns:
            Tuple of (bbox_xyxy, annotated_frame) where:
                - bbox_xyxy: numpy array [x1, y1, x2, y2] or None if lost
                - annotated_frame: frame with tracking annotation (or original if annotate=False)
        """
        if not self._is_tracking or self._tracker is None:
            return None, frame

        ok, bbox = self._tracker.update(frame)

        if not ok:
            if debug:
                print("[UAVTracker] Tracker update failed.")
            return None, frame

        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        h_img, w_img = frame.shape[:2]

        # Check loss conditions
        if self._is_lost(x1, y1, x2, y2, w_img, h_img, w * h, debug):
            return None, frame

        # Update current bbox
        self._current_bbox_xyxy = np.array([x1, y1, x2, y2], dtype=int)

        if not annotate:
            return self._current_bbox_xyxy, frame

        # Annotate frame
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            annotated_frame,
            "UAV (tracking)",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        return self._current_bbox_xyxy, annotated_frame

    def _is_lost(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        w_img: int,
        h_img: int,
        area: float,
        debug: bool = False
    ) -> bool:
        """
        Check if the tracked UAV should be considered lost.

        Args:
            x1, y1, x2, y2: Bounding box coordinates
            w_img, h_img: Image dimensions
            area: Bounding box area
            debug: Whether to print debug information

        Returns:
            True if UAV is lost, False otherwise
        """
        # 1) Bbox completely out of frame
        if x2 < 0 or y2 < 0 or x1 > w_img or y1 > h_img:
            if debug:
                print("[UAVTracker] Bbox completely out of frame -> lost.")
            return True

        # 2) Bbox touching/exceeding frame borders with margin
        if (
            x1 < -self.border_margin_px or
            y1 < -self.border_margin_px or
            x2 > w_img + self.border_margin_px or
            y2 > h_img + self.border_margin_px
        ):
            if debug:
                print("[UAVTracker] Bbox exceeds border margin -> lost.")
            return True

        # 3) Bbox too small
        if area < self.min_bbox_area:
            if debug:
                print(f"[UAVTracker] Bbox too small (area={area}) -> lost.")
            return True

        # 4) Bbox too large (relative to frame)
        img_area = w_img * h_img
        if area > self.max_bbox_area_ratio * img_area:
            if debug:
                print(f"[UAVTracker] Bbox too large (ratio={area/img_area:.2f}) -> lost.")
            return True

        return False
