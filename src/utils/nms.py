import numpy as np

from src.utils.my_types import BBox


def nms(bboxes: list[BBox], threshold: float = 0.) -> list[BBox]:
    """Apply Non-Maximum Suppression on the given bounding boxes.

    Args:
        bboxes: The list of bounding boxes
        threshold: Overlap threshold.

    Returns:
        A list of bounding boxes with the overlapping ones removed.
    """
    bboxes = np.asarray(bboxes)

    # Sort the bounding boxes by ascending area
    areas = bboxes[:, 2] * bboxes[:, 3]  # Area for each bbox
    sorting_idices = areas.argsort()
    areas = areas[sorting_idices]
    bboxes = bboxes[sorting_idices]

    kept_bboxes: list[BBox] = []
    while len(bboxes) > 0:
        # Keep biggest element and remove it from the input array.
        current_bbox, current_area = bboxes[-1], areas[:-1]
        kept_bboxes.append(BBox(*current_bbox))
        bboxes, areas = bboxes[:-1], areas[:-1]

        # Find the coordinates of the intersection boxes
        inter_x1 = np.maximum(bboxes[:, 0], current_bbox[0])
        inter_y1 = np.maximum(bboxes[:, 1], current_bbox[1])
        inter_x2 = np.minimum(bboxes[:, 0] + bboxes[:, 2], current_bbox[0] + current_bbox[2])
        inter_y2 = np.minimum(bboxes[:, 1] + bboxes[:, 3], current_bbox[1] + current_bbox[3])

        # Compute the area of the intersection boxes (0 if no intersection)
        inter_width = np.maximum(inter_x2 - inter_x1, 0)
        inter_height = np.maximum(inter_y2 - inter_y1, 0)
        inter_area = inter_width * inter_height

        union = areas + current_area - inter_area
        iou = inter_area / union

        bboxes, areas = bboxes[iou <= threshold], areas[iou <= threshold]

    return kept_bboxes
