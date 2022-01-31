"""
Author: John O'Neil
Email: oneil.john@gmail.com
Origin: https://github.com/johnoneil/MangaTextDetection/blob/master/connected_components.py
DATE: Saturday, August 10th 2013
      Revision: Thursday, August 15th 2013

  Connected component generation and manipulation utility functions.
"""
from typing import Optional

import cv2
import numpy as np
import scipy.ndimage


def draw_bounding_boxes(img: np.ndarray,
                        connected_components: list[tuple[slice, slice]],
                        min_size: Optional[int] = None,
                        color: int | tuple[int, int, int] = (0, 0, 255),
                        line_size: int = 2) -> np.ndarray:
    """Draw the connected components as bounding boxes on the image.

    Args:
        img: The image to draw on.
        connected_components: The connected components to draw (each component being a tuple of slices)
        min_size: Components smaller that thise will be filtered out and not drawn.
        color: Color of the bounding boxes (int for grayscale, tuple for color)
        line_size:  (use -1 for filled bounding boxes)

    Returns:
        The image with the bounding boxes on it.
    """
    draw_img = img.copy()
    for component in connected_components:
        if min_size is not None and bbox_area(component)**0.5 < min_size:
            continue
        ys, xs = component
        cv2.rectangle(draw_img, (xs.start, ys.start), (xs.stop, ys.stop), color, line_size)
    return draw_img


def components_to_bboxes(connected_components: list[tuple[slice, slice]]) -> list[tuple[int, int, int, int]]:
    """Converts components (slices) to bounding boxes (tuples).

    Args:
        connected_components: A list of components (a component is a tuple of slices)

    Returns:
        A list of 4 ints tuples (top left, top, width, height)
    """
    bboxes = []
    for component in connected_components:
        ys, xs = component
        bboxes.append((xs.start, ys.start, xs.stop-xs.start, ys.stop-ys.start))
    return bboxes


def get_connected_components(img: np.ndarray) -> list[tuple[slice, slice]]:
    """Get the connected components in the given image.

    Args:
        img (np.ndarray): The image to process.

    Returns:
        A list of tuples. Each tuple contains two slices that correspond to the bounds of the corresponding object.
    """
    # Generate a 3x3 matrix of Trues, i.e. a structuring element that will consider features connected
    # even if they touch diagonally. This results in a slightly smaller number of features
    # structure = scipy.ndimage.morphology.generate_binary_structure(2, 2)
    # label, num_features = scipy.ndimage.measurements.label(image, structure=structure)

    label, num_features = scipy.ndimage.measurements.label(img)
    objects_slices = scipy.ndimage.measurements.find_objects(label)
    return objects_slices


def bbox_area(bbox: tuple[slice, slice]) -> int:
    height = bbox[0].stop - bbox[0].start
    width = bbox[1].stop - bbox[1].start
    # assert width > 0 and height > 0, f"Bounding box has {width=}, {height=}. They should be > 0"
    return height * width


def get_cc_average_size(img: np.ndarray, minimum_area: int = 3, maximum_area: int = 100) -> float:
    """Compute the connected components of the given image, and return their average size after filtering by area.

    Args:
        img: The image to process.
        minimum_area: Components smaller than this value are discarded. TODO: Not used as an area value...
        maximum_area: Components bigger than this value are discarded.

    Returns:
        The median value of the areas within the given bounds.
    """
    components = get_connected_components(img)
    sorted_components = sorted(components, key=bbox_area)
    # sorted_components = sorted(components,key=lambda x:area_nz(x,binary))
    areas = np.zeros(img.shape)
    for component in sorted_components:
        # If an area has already been set within the current component, then skip. (i.e. no overwriting)
        # Since the components are sorted from smallest to biggest, this means that we only keep the smaller ones.
        # (For example individual letters vs whole panel)
        if np.amax(areas[component]) > 0:
            continue
        # TODO: check if the area value used has a big influence.
        # Take the sqrt of the area of the bounding box
        areas[component] = bbox_area(component)**0.5
        # Alternate implementation where we just use area of black pixels in cc
        # areas[component] = area_nz(component, binary)

    # Only keep the areas values that are within the desired range.
    # (by default connected components between 3 and 100 pixels on a side (text sized))
    kept_areas = areas[(areas > minimum_area) & (areas < maximum_area)]
    if len(kept_areas) == 0:
        return 0
    # Lastly take the median  # TODO: compare with average. Compare with median of set.
    return np.median(kept_areas)


def form_mask(img: np.ndarray, max_size: float, min_size: float) -> np.ndarray:
    """Create a binary (1 & 0s) mask based on the image and its connected components.

    This keep only the elements of the image that belong to a connected components whose size is within
    the given bounds.
    TODO: This function also assume that the text is white on black (it is the case because of the inv binary threshold)

    Args:
        img (np.ndarray): The image to process.
        max_size (float): Components bigger than this size are discarded (i.e. not included in the mask)
        min_size (float): Same as above but with smaller components.

    Returns:
        The binary mask.
    """
    components = get_connected_components(img)
    sorted_components = sorted(components, key=bbox_area)
    mask = np.zeros(img.shape, np.uint8)
    for component in sorted_components:
        if bbox_area(component)**.5 < min_size or bbox_area(component)**.5 > max_size:
            continue
        mask[component] = img[component] > 0
    return mask
