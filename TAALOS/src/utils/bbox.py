import numpy as np


def get_bbox_center(bbox_xyxy: np.ndarray) -> tuple[float, float]:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox_xyxy: Bounding box in [x1, y1, x2, y2] format

    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return center_x, center_y