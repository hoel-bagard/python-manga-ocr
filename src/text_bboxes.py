import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import scipy.ndimage

from src.utils.connected_components import (
    components_to_bboxes,
    form_mask,
    get_cc_average_size,
    get_connected_components
)
from src.utils.logger import create_logger
from src.utils.misc import show_img
from src.utils.rlsa import rlsa


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
                         average_size: float,
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
        average_size: The average size of the component (i.e. object blocks) on the image.
        logger: Logger use to print things.
        display_images: If True, some images might get displayed.

    Returns:
        An mask with the same shape as the input image. (white where text is, black elsewhere)
        The mask is composed of rectangles, each rectangle corresponding to a text zone.
    """
    vsv = 1*average_size  # TODO: Put those magic numbers in the config
    hsv = 1*average_size
    height, width = cleaned_img.shape[:2]
    logger.debug(f"Applying run length smoothing with vertical threshold {vsv:.2f} and horizontal threshold {hsv:.2f}")

    rlsa_result = rlsa(cleaned_img, hsv, vsv)
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
        # cv2.rectangle(text, (component[1].start, component[0].start), (component[1].stop, component[0].stop), 255, -1)
        text[component[0].start:component[0].stop, component[1].start:component[1].stop] = 255

    print(display_images)
    print(logger.getEffectiveLevel())
    if logger.getEffectiveLevel() == logging.DEBUG and display_images:
        components_img = cleaned_img.copy()
        [cv2.rectangle(components_img, (c[1].start, c[0].start), (c[1].stop, c[0].stop), 127, 4) for c in components]
        show_img(cv2.hconcat([cleaned_img, rlsa_result, components_img, text]), "Input, RLSA result, components, text")

    return text


def get_text_bboxes(img: np.ndarray,
                    logger: logging.Logger,
                    min_scale: float = 0.25,
                    max_scale: float = 4.,
                    display_images: bool = False) -> list[tuple[int, int, int, int]]:
    """Return the bounding boxes corresponding to the blocks of text on the image.

    Args:
        img: The image to process.
        logger: A logger, mostly used to print debug information (also shows images in debug mode).
        max_scale: Used to filter out components.
        min_scale: Used to filter out components.
        display_images: If True, some images might get displayed.

    Returns:
        A list of tuples. Each tuple has 4 ints forming a bounding box: (top left, top, width, height).
    """
    height, width = img.shape[:2]
    logger.debug(f"Processing {height}x{width} image, looking for text bounding boxes.")

    # Create gaussian filtered and unfiltered binary images
    binary_threshold = 230  # TODO: Config
    logger.debug(f"Binarizing images with threshold value of {binary_threshold}")
    _, binary = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY_INV)

    binary_average_size = get_cc_average_size(binary)
    logger.debug(f"Average cc size for binaryized grayscale image is {binary_average_size:.2f}")

    sigma = 1.5  # (0.8/676.0)*float(height)-0.9
    logger.debug(f"Applying Gaussian filter with sigma (std dev) of {sigma:.2f}")
    gaussian_filtered = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    _, gaussian_binary = cv2.threshold(gaussian_filtered, binary_threshold, 255, cv2.THRESH_BINARY_INV)

    # Draw out statistics on average connected component size in the rescaled, binary image
    average_size = get_cc_average_size(gaussian_binary)
    logger.debug(f"Binarized Gaussian filtered image average cc size: {average_size:.2f}")
    max_size = average_size*max_scale
    min_size = average_size*min_scale

    # Create a mask based on the connected components's non zero values, filtered by size
    mask = form_mask(gaussian_binary, max_size, min_size)
    # Secondary mask is formed from the convex hulls around the canny edges (masked by the previous mask)
    canny_mask = get_canny_hulls_mask(gaussian_filtered, mask=mask)
    # Final mask is size filtered connected components on canny mask
    final_mask = form_mask(canny_mask, max_size, min_size)

    # Apply mask and return images
    cleaned = cv2.bitwise_not(final_mask * binary)
    text_only = cleaned_to_text_mask(cleaned, average_size, logger, display_images)

    # TODO: That was actually used.
    # If desired, suppress furigana characters (which interfere with OCR)
    # suppress_furigana = arg.boolean_value('furigana')
    # if suppress_furigana:
    #     logger.debug("Attempting to suppress furigana characters which interfere with OCR.")
    #     furigana_mask = furigana.estimate_furigana(cleaned, text_only)
    #     furigana_mask = np.array(furigana_mask==0,'B')
    #     cleaned = cv2.bitwise_not(cleaned)*furigana_mask
    #     cleaned = cv2.bitwise_not(cleaned)
    #     text_only = cleaned2segmented(cleaned, average_size)

    # TODO: That was actually used.
    # (text_like_areas, nontext_like_areas) = filter_text_like_areas(img, segmentation=text_only,
    #                                                                average_size=average_size)
    # logger.debug(f"{len(text_like_areas)} potential text areas have been detected in total.")
    # text_only = np.zeros(img.shape)
    # text_only = draw_bounding_boxes(text_only, text_like_areas, color=(255), line_size=-1)

    if logger.getEffectiveLevel() == logging.DEBUG and display_images:
        debug_img = np.zeros((height, width, 3), np.uint8)
        debug_img[:, :, 0] = img
        debug_img[:, :, 1] = text_only
        debug_img[:, :, 2] = text_only
        show_img(debug_img)

    final_components = get_connected_components(text_only)
    final_bboxes = components_to_bboxes(final_components)
    return final_bboxes


def main():
    parser = ArgumentParser(description="Manga page cleaning script. Run with 'python -m src.clean_image <path>'.")
    parser.add_argument("img_path", type=Path, help="Path to the dataset")
    parser.add_argument("--display_images", "-d", action="store_true", help="Displays some debug images.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    img_path: Path = args.img_path
    display_images: bool = args.display_images
    verbose_level: str = args.verbose_level
    logger = create_logger("Manga cleaner", verbose_level=verbose_level)

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    bboxes = get_text_bboxes(img, logger, display_images=display_images)
    for left, top, width, height in bboxes:
        if width * height < 5000:
            continue
        cv2.rectangle(img, (left, top), (left+width, top+height), 127, -1)  # Last param: 3 for contour, -1 for filled
    show_img(img)


if __name__ == "__main__":
    main()
