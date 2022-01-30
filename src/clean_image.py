import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import scipy.ndimage

from src.utils.connected_components import get_cc_average_size, get_connected_components, form_mask
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


def cleaned2segmented(cleaned: np.ndarray, average_size: float, logger: logging.Logger):
    vertical_smoothing_threshold = 0.75*average_size  # Note: the 0.75 used to be in the config
    horizontal_smoothing_threshold = 0.75*average_size
    height, width = cleaned.shape[:2]
    logger.debug(f"Applying run length smoothing with vertical threshold {vertical_smoothing_threshold} "
                 f"and horizontal threshold {horizontal_smoothing_threshold}")

    run_length_smoothed = rls.RLSO(cv2.bitwise_not(cleaned), vertical_smoothing_threshold,
                                   horizontal_smoothing_threshold)
    components = get_connected_components(run_length_smoothed)
    text = np.zeros((height, width), np.uint8)
    for component in components:
        seg_thresh = 1

        def segment_into_lines(img, component, min_segment_threshold=1):
            (ys, xs) = component[:2]
            vertical = []
            start_col = xs.start
            for col in range(xs.start, xs.stop):
                count = np.count_nonzero(img[ys.start:ys.stop, col])
                if count <= min_segment_threshold or col == (xs.stop):
                    if start_col >= 0:
                        vertical.append((slice(ys.start, ys.stop), slice(start_col, col)))
                        start_col = -1
                elif start_col < 0:
                    start_col = col

        aspect, v_lines, h_lines = segment_into_lines(cv2.bitwise_not(cleaned), component,
                                                      min_segment_threshold=seg_thresh)
        if len(v_lines) < 2 and len(h_lines) < 2:
            continue

        # TODO: Wouldn't it be possible to just use the slice to set the values to 255 ?
        cv2.rectangle(text, (component[1].start, component[0].start), (component[1].stop, component[0].stop), 255, -1)
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
    logger.debug(f"Applying Gaussian filter with sigma (std dev) of {sigma}")
    gaussian_filtered = scipy.ndimage.gaussian_filter(img, sigma=sigma)

    _, gaussian_binary = cv2.threshold(gaussian_filtered, binary_threshold, 255, cv2.THRESH_BINARY_INV)

    # Draw out statistics on average connected component size in the rescaled, binary image
    average_size = get_cc_average_size(gaussian_binary)
    logger.debug(f"Binarized Gaussian filtered image average cc size: {average_size}")
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
    text_only = cleaned2segmented(cleaned, average_size)

    # If desired, suppress furigana characters (which interfere with OCR)
    # suppress_furigana = arg.boolean_value('furigana')
    # if suppress_furigana:
    #     logger.debug("Attempting to suppress furigana characters which interfere with OCR.")
    #     furigana_mask = furigana.estimate_furigana(cleaned, text_only)
    #     furigana_mask = np.array(furigana_mask==0,'B')
    #     cleaned = cv2.bitwise_not(cleaned)*furigana_mask
    #     cleaned = cv2.bitwise_not(cleaned)
    #     text_only = cleaned2segmented(cleaned, average_size)

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
