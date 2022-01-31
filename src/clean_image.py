import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import scipy.ndimage

from src.utils.connected_components import get_cc_average_size, get_connected_components, form_mask
from src.utils.rlsa import rlsa
from src.utils.logger import create_logger
from src.utils.misc import show_img


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
        if count <= min_threshold or row == (y_slice.stop):
            if start_row >= 0:
                lines.append((slice(start_row, row), slice(x_slice.start, x_slice.stop)))
                start_row = -1
        elif start_row < 0:
            start_row = row
    return lines


def cleaned2segmented(cleaned: np.ndarray, average_size: float, logger: logging.Logger):
    vsv = 0.75*average_size  # Note: the 0.75 used to be in the config
    hsv = 0.75*average_size
    height, width = cleaned.shape[:2]
    logger.debug(f"Applying run length smoothing with vertical threshold {vsv:.2f} and horizontal threshold {hsv:.2f}")

    rlsa_result = rlsa(cleaned, hsv, vsv)
    components = get_connected_components(cv2.bitwise_not(rlsa_result))

    text = np.zeros((height, width), np.uint8)
    for component in components:
        seg_thresh = 1
        h_lines = detect_lines(cv2.bitwise_not(cleaned), component[1], component[0], seg_thresh)
        v_lines = detect_lines(cv2.bitwise_not(cleaned).T, component[0], component[1], seg_thresh)

        # Filter out components that are composed of one block of black.
        # These are often drawings that got through the previous steps.
        if len(v_lines) < 2 and len(h_lines) < 2:
            continue

        # TODO: Wouldn't it be possible to just use the slice to set the values to 255 ?
        cv2.rectangle(text, (component[1].start, component[0].start), (component[1].stop, component[0].stop), 255, -1)

    if logger.getEffectiveLevel() == logging.DEBUG:
        components_img = cleaned.copy()
        [cv2.rectangle(components_img, (c[1].start, c[0].start), (c[1].stop, c[0].stop), 127, 4) for c in components]
        show_img(cv2.hconcat([cleaned, rlsa_result, components_img, text]), "Input, RLSA result, components and text")

    return text


# def filter_text_like_areas(img, segmentation, average_size):
#   #see if a given rectangular area (2d slice) is very text like
#   #First step is to estimate furigana like elements so they can be masked
#   furigana_areas = furigana.estimate_furigana(img, segmentation)
#   furigana_mask = np.array(furigana_areas==0,'B')

#   #binarize the image, clean it via the segmentation and remove furigana too
#   binary_threshold = arg.integer_value('binary_threshold',default_value=defaults.BINARY_THRESHOLD)
#   if arg.boolean_value('verbose'):
#     print('binarizing images with threshold value of ' + str(binary_threshold))
#   binary = clean.binarize(img,threshold=binary_threshold)

#   binary_average_size = cc.average_size(binary)
#   if arg.boolean_value('verbose'):
#     print('average cc size for binaryized grayscale image is ' + str(binary_average_size))
#   segmentation_mask = np.array(segmentation!=0,'B')
#   cleaned = binary * segmentation_mask * furigana_mask
#   inv_cleaned = cv2.bitwise_not(cleaned)

#   areas = cc.get_connected_components(segmentation)
#   text_like_areas = []
#   nontext_like_areas = []
#   for area in areas:
#     #if area_is_text_like(cleaned, area, average_size):
#     if text_like_histogram(cleaned, area, average_size):
#       text_like_areas.append(area)
#     else:
#       nontext_like_areas.append(area)

#   return (text_like_areas, nontext_like_areas)


def segment_image(img: np.ndarray,
                  logger: logging.Logger,
                  max_scale: float = 4.,
                  min_scale: float = 0.15):
    height, width = img.shape[:2]
    logger.info(f"Segmenting {height}x{width} image.")

    # Create gaussian filtered and unfiltered binary images
    binary_threshold = 190
    logger.debug(f"Binarizing images with threshold value of {binary_threshold}")
    _, binary = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY_INV)

    binary_average_size = get_cc_average_size(binary)
    logger.debug(f"Average cc size for binaryized grayscale image is {binary_average_size:.2f}")

    # The necessary sigma needed for Gaussian filtering (to remove screentones and other noise) seems
    # to be a function of the resolution the manga was scanned at (or original page size, I'm not sure).
    # Assuming "normal" page size for a phonebook style Manga is 17.5cmx11.5cm (6.8x4.5in).
    # A scan of 300dpi will result in an image about 1900x1350, which requires a sigma of 1.5 to 1.8.
    # I'm encountering many smaller images that may be nonstandard scanning dpi values or just smaller magazines.
    # Haven't found hard info on this yet. They require sigma values of about 0.5 to 0.7.
    # I'll therefore (for now) just calculate required (nonspecified) sigma as a linear function of vertical
    # image resolution.
    sigma = (0.8/676.0)*float(height)-0.9
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
    text_only = cleaned2segmented(cleaned, average_size, logger)

    # If desired, suppress furigana characters (which interfere with OCR)
    # suppress_furigana = arg.boolean_value('furigana')
    # if suppress_furigana:
    #     logger.debug("Attempting to suppress furigana characters which interfere with OCR.")
    #     furigana_mask = furigana.estimate_furigana(cleaned, text_only)
    #     furigana_mask = np.array(furigana_mask==0,'B')
    #     cleaned = cv2.bitwise_not(cleaned)*furigana_mask
    #     cleaned = cv2.bitwise_not(cleaned)
    #     text_only = cleaned2segmented(cleaned, average_size)

    # TODO: comment below might not be relevant. Careful
#     # we've now broken up the original bounding box into possible vertical and horizontal lines.
#     # We now wish to:
#     # 1) Determine if the original bounding box contains text running V or H
#     # 2) Eliminate all bounding boxes (don't return them in our output lists) that
#     #    we can't explicitly say have some "regularity" in their line width/heights
#     # 3) Eliminate all bounding boxes that can't be divided into v/h lines at all(???)
#     # also we will give possible vertical text runs preference as they're more common
    (text_like_areas, nontext_like_areas) = filter_text_like_areas(img, segmentation=text_only,
                                                                   average_size=average_size)
    logger.debug(f"{len(text_like_areas)} potential textl areas have been detected in total.")
    text_only = np.zeros(img.shape)
    cc.draw_bounding_boxes(text_only, text_like_areas,color=(255),line_size=-1)

    # if arg.boolean_value('debug'):
    #     text_only = 0.5*text_only + 0.5*img
    #     # text_rows = 0.5*text_rows+0.5*gray
    #     # text_colums = 0.5*text_columns+0.5*gray

    segmented_image = np.zeros((height, width, 3), np.uint8)
    segmented_image[:, :, 0] = img
    segmented_image[:, :, 1] = text_only
    segmented_image[:, :, 2] = text_only
    return segmented_image


# def get_bboxes(img: np.ndarray, logger: logging.Logger):
#     # binary_threshold = 190
#     # logger.debug(f"Binarizing with threshold value of {binary_threshold}")
#     # _, binary = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY_INV )

#     segmented_image = seg.segment_image(gray)
#     segmented_image = segmented_image[:, :, 2]
#     components = cc.get_connected_components(segmented_image)
#     cc.draw_bounding_boxes(img,components,color=(255,0,0),line_size=2)


def main():
    parser = ArgumentParser(description="Manga page cleaning script. Run with 'python -m src.clean_image <path>'.")
    parser.add_argument("img_path", type=Path, help="Path to the dataset")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    img_path: Path = args.img_path
    verbose_level: str = args.verbose_level
    logger = create_logger("Manga cleaner", verbose_level=verbose_level)

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    # show_img(img)

    segment_image(img, logger)
    # get_bboxes(img, logger)


if __name__ == "__main__":
    main()
