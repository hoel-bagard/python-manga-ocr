import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import scipy.ndimage
from rlsa import rlsa

from config.generation_config import GenerationConfig
from src.utils.connected_components import (
    components_to_bboxes,
    form_mask,
    get_cc_average_size,
    get_connected_components
)
from src.utils.logger import create_logger
from src.utils.misc import show_img
from src.utils.my_types import BBox


def filter_bbox_size(bboxes: list[BBox], min_area: int = 5000) -> list[BBox]:
    """Filter out all the bounding boxes whose area is bellow the given threshold."""
    for bbox in reversed(bboxes):
        *_, width, height = bbox
        if width * height < min_area:
            bboxes.remove(bbox)
    return bboxes


def get_canny_hulls_mask(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Get the canny edges of the given image and draw blocks around them to form a mask.

    Args:
        img: The image to process.
        mask: Optional, binary mask applied after the canny edges detection.

    Returns:
        The resulting mask
    """
    edges = cv2.Canny(img, 128, 255, apertureSize=3)
    mask = mask*edges if mask is not None else edges
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hulls_mask = np.zeros(img.shape, np.uint8)
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(hulls_mask, [hull], 0, 255, -1)

    return hulls_mask


def detect_lines(img: np.ndarray,
                 x_slice: slice,
                 y_slice: slice,
                 min_threshold: int = 1) -> list[tuple[slice, slice]]:
    """Detect lines in the given area as horizontal rows of non-zero pixels.

    Note: This function can also be used to get the columns by passing in the transpose of the image
          and inverting the x and y slices.

    This function counts the number of black pixels on each line, and when that number goes bellow the given
    threshold, it considers that the current line ended. It then continues to go down the image, looking for the
    next line (i.e. when the number of black pixels gets superior to the threshold again)

    Args:
        img (np.ndarray): The image to process. It is expected to be black text on a white background.
        x_slice (slice): The horizontal limits of the area to process.
        y_slice (slice): The vertical limits of the area to process.
        min_threshold (int): The threshold (in number of pixel) used to separate 2 lines.

    Returns:
        A list containing the bounding boxes (as slices) for all the lines detected
    """
    lines = []
    start_row = y_slice.start  # Used both as the starting row if currently on a line, or as a flag.
    for row in range(y_slice.start, y_slice.stop):
        count = np.count_nonzero(img[row, x_slice.start:x_slice.stop])
        if count <= min_threshold or row == y_slice.stop-1:
            if start_row >= 0:
                lines.append((slice(start_row, row), slice(x_slice.start, x_slice.stop)))
                start_row = -1
        elif start_row < 0:
            start_row = row
    return lines


def cleaned_to_text_mask(cleaned_img: np.ndarray,
                         hsv: int,
                         vsv: int,
                         logger: logging.Logger,
                         display_images: bool = False) -> np.ndarray:
    """Takes in a cleaned image and returns a mask of the (likely) text boxes.

    - First apply RLSA to the image to connect the letters / objects.
    - Then get the connected components (since the letters have been linked,
      connected component corresponds to a block of text).
    - Finally, filter out components that don't look like text (text is expected to be composed of lines/blocks of text)

    Args:
        cleaned_img: The image to process. It should be a black and white image that has already been
                     preprocessed so that most of the background / noise has been removed.
        hsv: The horizontal threshold to use in the RLSA.
        vsv: The vertical threshold to use in the RLSA.
        logger: Logger use to print things.
        display_images: If True, some images might get displayed.

    Returns:
        An mask with the same shape as the input image. (white where text is, black elsewhere)
        The mask is composed of rectangles, each rectangle corresponding to a text zone.
    """
    height, width = cleaned_img.shape[:2]
    logger.debug(f"Applying run length smoothing with vertical threshold {vsv:.2f} and horizontal threshold {hsv:.2f}")

    rlsa_result = rlsa(cleaned_img, hsv, vsv, hsv//10)
    components = get_connected_components(cv2.bitwise_not(rlsa_result))

    text = np.zeros((height, width), np.uint8)
    for component in components:
        seg_thresh = 1
        h_lines = detect_lines(cv2.bitwise_not(cleaned_img), component[1], component[0], seg_thresh)
        v_lines = detect_lines(cv2.bitwise_not(cleaned_img).T, component[0], component[1], seg_thresh)

        # Filter out components that are composed of one block of black.
        # These are often drawings that got through the previous steps.
        if len(v_lines) < 2 and len(h_lines) < 2:
            continue

        # Mask the component's location.
        text[component[0].start:component[0].stop, component[1].start:component[1].stop] = 255

    if logger.getEffectiveLevel() == logging.DEBUG and display_images:
        components_img = cleaned_img.copy()
        [cv2.rectangle(components_img, (c[1].start, c[0].start), (c[1].stop, c[0].stop), 127, 4) for c in components]
        show_img(cv2.hconcat([cleaned_img, rlsa_result, components_img, text]), "Input, RLSA result, components, text")

    return text


def get_text_bboxes(img: np.ndarray,
                    logger: logging.Logger,
                    config: GenerationConfig,
                    display_images: bool = False) -> list[tuple[int, int, int, int]]:
    """Return the bounding boxes corresponding to the blocks of text on the image.

    Args:
        img: The image to process.
        logger: A logger, mostly used to print debug information (also shows images in debug mode).
        config: The config with the values to use for diverse steps.
        display_images: If True, some images might get displayed.

    Returns:
        A list of tuples. Each tuple has 4 ints forming a bounding box: (top left, top, width, height).
    """
    height, width = img.shape[:2]
    logger.debug(f"Processing {height}x{width} image, looking for text bounding boxes.")

    # Create gaussian filtered and unfiltered binary images
    logger.debug(f"Binarizing images with threshold value of {config.binary_threshold}")
    _, binary = cv2.threshold(img, config.binary_threshold, 255, cv2.THRESH_BINARY_INV)

    binary_average_size = get_cc_average_size(binary)
    logger.debug(f"Average cc size for binaryized grayscale image is {binary_average_size:.2f}")

    sigma = 1.5  # (0.8/676.0)*height0.9
    logger.debug(f"Applying Gaussian filter with sigma (std dev) of {sigma:.2f}")
    gaussian_filtered = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    _, gaussian_binary = cv2.threshold(gaussian_filtered, config.binary_threshold, 255, cv2.THRESH_BINARY_INV)

    # Draw out statistics on average connected component size in the rescaled, binary image
    average_area = get_cc_average_size(gaussian_binary)
    logger.debug(f"Binarized Gaussian filtered image average cc area: {average_area:.2f}")
    max_size = math.sqrt(average_area)*config.max_scale
    min_size = math.sqrt(average_area)*config.min_scale

    # Create a mask based on the connected components's non zero values, filtered by size
    mask = form_mask(gaussian_binary, max_size, min_size)
    # Secondary mask is formed from the convex hulls around the canny edges (masked by the previous mask)
    canny_mask = get_canny_hulls_mask(gaussian_filtered, mask=mask)
    # Final mask is size filtered connected components on canny mask
    final_mask = form_mask(canny_mask, max_size, min_size)

    # Apply mask and return images
    cleaned = cv2.bitwise_not(final_mask * binary)
    text_only = cleaned_to_text_mask(cleaned, config.hsv, config.vsv, logger, display_images)

    if logger.getEffectiveLevel() == logging.DEBUG and display_images:
        debug_img = np.zeros((height, width, 3), np.uint8)
        debug_img[:, :, 0] = img
        debug_img[:, :, 1] = text_only
        debug_img[:, :, 2] = text_only
        show_img(debug_img)

    final_components = get_connected_components(text_only)
    bboxes = components_to_bboxes(final_components)
    final_bboxes = filter_bbox_size(bboxes, config.min_bbox_area)
    return final_bboxes


def filter_bubbles(img: npt.NDArray[np.uint8],
                   bboxes: list[BBox],
                   config: GenerationConfig,
                   display_imgs: bool = False) -> tuple[npt.NDArray[np.uint8], list[BBox]]:
    """Given an image and a list of likely text bounding boxes, try to remove some false detection.

    The idea here is to remove the "text" that has been detected, and then to look for other components within the
    text's container. Usually a text bubble contains only text, therefore if there are other things within the
    container, then it was probably not a text bubble.

    Args:
        img: The original image.
        bboxes: The text bouding boxes candidates.
        config: The config with the values to use for diverse steps.
        display_imgs: Use for debug, display some intermediate results.

    Returns:
        The list of kept (and expanded) bounding boxes, along with the image with the bubbles' inside removed.
    """
    input_img = img.copy()
    # Threshold to get some clear boundaries for the text bubbles
    _, img = cv2.threshold(img, 252, 255, cv2.THRESH_BINARY)

    # Remove the detected "text" from the bubble.
    # Note: I used rlsa and connected components instead of simply the bounding box because otherwise the bottom corners
    #       of the bbox can pierce the bubble's boundary, leading to it being discarded later on.
    for left, top, width, height in bboxes:
        img_patch = img[top:top+height, left:left+width].copy()
        img_patch = rlsa(img_patch, config.hsv//2, config.vsv, config.hsv//4)
        img_patch = cv2.bitwise_not(img_patch)
        nb_labels, img_labels = cv2.connectedComponents(img_patch)

        # With the opencv cc, there is always 0 for the background and then one int per cc.
        # If there is more than one cc, then keep only the biggest.
        idx = np.argmax([len(img_labels[img_labels == i]) for i in range(1, nb_labels)]) + 1
        img[top:top+height, left:left+width] = np.where(img_labels != idx, img[top:top+height, left:left+width], 255)

    # There's some noise around where the text was, try to remove a bit of it without creating holes in the border
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # if display_imgs:
    #     show_img(img, "Cleaned image")

    # Floodfill each potential bubble and get the resulting mask.
    mask = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=np.uint8)
    for left, top, width, height in bboxes:
        center = left+width//2, top+height//2
        cv2.floodFill(img, mask, center, 125)

    # Remove little imperfections in the mask.
    mask = 255*mask
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=4)
    mask = mask[1:-1, 1:-1]  # Remove padding

    if display_imgs:
        show_img(img, "Floodfilled image")
        # show_img(mask, "Flood mask")

    # Get the contours of the objects in the mask.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)

    # Remove all the contours that have children or are children
    kept_contours = []
    mask_contours = np.zeros_like(input_img)
    for i, (*_, child, hierarchy_lvl) in enumerate(hierarchy[0]):
        if child == -1 and hierarchy_lvl == -1:
            kept_contours.append(contours[i])
            cv2.drawContours(mask_contours, contours, i, 100, -1, cv2.LINE_8, hierarchy, 0)
            cv2.drawContours(mask_contours, contours, i, 255, 2, cv2.LINE_8, hierarchy, 0)
    if display_imgs:
        debug_contours = np.zeros_like(input_img)
        for i in range(len(contours)):
            cv2.drawContours(debug_contours, contours, i, 255, 2, cv2.LINE_8, hierarchy, 0)
        # show_img(debug_contours, "All contours")
        show_img(mask_contours, "Kept contours")

    # Match each kept contour to its text bbox. Then progressively enlarge the bbox until it fits the bubble.
    # Note: I could start from the contour's centroid instead (easy to get with the moments), but that would likely
    #       result in more square-ish bboxes (i.e. worse fit to the bubble).

    # Match each kept contour to its text bbox. Then progressively enlarge the bbox until it fits the bubble.
    # Note: I could start from the contour's centroid instead (easy to get with the moments), but that would likely
    #       result in more square-ish bboxes (i.e. worse fit to the bubble).
    # Some small, garbage contours do not get matched with a bbox so some of the contours need to get removed.
    matched_contours: list[npt.NDArray[np.int32]] = []
    adjusted_text_bboxes: list[BBox] = []
    for contour in kept_contours:
        x, y, w, h = cv2.boundingRect(contour)
        for left, top, width, height in bboxes:
            center_x, center_y = left+width//2, top+height//2
            # If the center of a text bbox is within the bounding rect of a contour, then we consider that they match.
            if x <= center_x and center_x <= x+w and y <= center_y and center_y <= y+h:
                prev_bbox = (0, 0, 0, 0)
                while prev_bbox != (left, top, width, height):
                    prev_bbox = (left, top, width, height)
                    # Move left line
                    if (mask_contours[top:top+height, left] == 100).all():
                        left -= 1
                    # Move right line
                    if (mask_contours[top:top+height, left+width] == 100).all():
                        width += 1
                    # Move top line
                    if (mask_contours[top, left:left+width] == 100).all():
                        top -= 1
                    # Move bottom line
                    if (mask_contours[top+height, left:left+width] == 100).all():
                        height += 1
                adjusted_text_bboxes.append(BBox(left, top, width, height))
                matched_contours.append(contour)

    # Filter by area again. With a bigger value than for just the text box.
    # The idea is that even for a small text box (one or two charaters), the bubble is usually pretty big, and therefore
    # the expanded bbox should have a bigger area.
    # Whereas a small non text bbox will probably not expand much (an eye for example).
    final_text_bboxes: list[BBox] = []
    final_contours = []
    for bbox, contour in zip(adjusted_text_bboxes, matched_contours):
        if bbox.width * bbox.height > 3*config.min_bbox_area:
            final_text_bboxes.append(bbox)
            final_contours.append(contour)

    # Remove the inside of the bubbles from the image by using the contours, so that we can put anything there later.
    # Note: A contour can match multiple bounding boxes, therefore there might be duplicate contoursin final_contours.
    #         --> For example when two bubbles are "fused" (see blackjack ni yoroshiku vol1 page30 for an example).
    #       OpenCV's drawContours function does not handle duplicates when filling countours, hence the for loop.
    final_mask = np.zeros_like(input_img)
    for i in range(len(final_contours)):
        cv2.drawContours(final_mask, final_contours, i, 255, -1, cv2.LINE_8)
    cleaned_img = np.where(final_mask == 255, 255, input_img)

    if display_imgs:
        bbox_img = cleaned_img.copy()
        for bbox in bboxes:
            bbox_img = cv2.rectangle(bbox_img, (bbox.left, bbox.top), (bbox.left+bbox.width, bbox.top+bbox[3]), 122, 5)
        for bbox in final_text_bboxes:
            bbox_img = cv2.rectangle(bbox_img, (bbox.left, bbox.top), (bbox.left+bbox.width, bbox.top+bbox[3]), 0, 5)
        show_img(bbox_img, "Cleaned image with text bboxes")

    return cleaned_img, final_text_bboxes


def main():
    from argparse import ArgumentParser
    from config.generation_config import get_generation_config

    parser = ArgumentParser(description=("Script to clean the bubbles of text from manga pages and get their locations."
                                         "Run with 'python -m src.prepare_data <path>'."))
    parser.add_argument("data_path", type=Path, help="Path to the folder with the manga pages.")
    parser.add_argument("--display_images", "-d", action="store_true", help="Displays some debug images.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    display_images: bool = args.display_images
    verbose_level: str = args.verbose_level
    logger = create_logger("Manga cleaner", verbose_level=verbose_level)

    exts = [".jpg", ".png", ".bmp", ".ppm"]
    img_paths = list([p for p in data_path.rglob("*") if p.suffix in exts])
    nb_imgs = len(img_paths)

    for i, img_path in enumerate(img_paths, start=1):
        # clean_print(f"Processing image {img_path.name}    ({i}/{nb_imgs})", end="\r" if i != nb_imgs else "\n")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        config = get_generation_config()

        bboxes = get_text_bboxes(img, logger, config, display_images=display_images)
        cleaned_img, final_text_bboxes = filter_bubbles(img, bboxes, config, display_images)

        if display_images:
            for bbox in final_text_bboxes:
                cleaned_img = cv2.rectangle(cleaned_img, (bbox.left, bbox.top), (bbox.left+bbox.width, bbox.top+bbox[3]), 0, 5)
            show_img(cleaned_img, f"Result for img {img_path}")


if __name__ == "__main__":
    main()
