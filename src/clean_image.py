import logging
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from src.utils.connected_components import get_cc_average_size
from src.utils.logger import create_logger
from src.utils.misc import show_img


def segment_image(img: np.ndarray,
                  logger: logging.Logger,
                  max_scale: float = 4.,
                  min_scale: float = 0.15):
    h, w = img.shape[:2]

    logger.info(f"Segmenting {h}x{w} image.")

    # Create gaussian filtered and unfiltered binary images
    binary_threshold = 190
    logger.debug(f"Binarizing images with threshold value of {binary_threshold}")
    _, binary = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY_INV)

    binary_average_size = get_cc_average_size(binary)
    logger.debug(f"Average cc size for binaryized grayscale image is {binary_average_size:.2f}")
    exit()

    # The necessary sigma needed for Gaussian filtering (to remove screentones and other noise) seems
    # to be a function of the resolution the manga was scanned at (or original page size, I'm not sure).
    # Assuming 'normal' page size for a phonebook style Manga is 17.5cmx11.5cm (6.8x4.5in).
    # A scan of 300dpi will result in an image about 1900x1350, which requires a sigma of 1.5 to 1.8.
    # I'm encountering many smaller images that may be nonstandard scanning dpi values or just smaller
    # magazines. Haven't found hard info on this yet. They require sigma values of about 0.5 to 0.7.
    # I'll therefore (for now) just calculate required (nonspecified) sigma as a linear function of vertical
    # image resolution.
    sigma = (0.8/676.0)*float(h)-0.9
    sigma = arg.float_value('sigma',default_value=sigma)
    if arg.boolean_value('verbose'):
        print('Applying Gaussian filter with sigma (std dev) of ' + str(sigma))
    gaussian_filtered = scipy.ndimage.gaussian_filter(img, sigma=sigma)

    gaussian_binary = clean.binarize(gaussian_filtered,threshold=binary_threshold)

    #Draw out statistics on average connected component size in the rescaled, binary image
    average_size = cc.average_size(gaussian_binary)
    if arg.boolean_value('verbose'):
        print('Binarized Gaussian filtered image average cc size: ' + str(average_size))
    max_size = average_size*max_scale
    min_size = average_size*min_scale

    #primary mask is connected components filtered by size
    mask = cc.form_mask(gaussian_binary, max_size, min_size)

    #secondary mask is formed from canny edges
    canny_mask = clean.form_canny_mask(gaussian_filtered, mask=mask)

    #final mask is size filtered connected components on canny mask
    final_mask = cc.form_mask(canny_mask, max_size, min_size)

    #apply mask and return images
    cleaned = cv2.bitwise_not(final_mask * binary)
    text_only = cleaned2segmented(cleaned, average_size)

    #if desired, suppress furigana characters (which interfere with OCR)
    suppress_furigana = arg.boolean_value('furigana')
    if suppress_furigana:
        if arg.boolean_value('verbose'):
            print('Attempting to suppress furigana characters which interfere with OCR.')
        furigana_mask = furigana.estimate_furigana(cleaned, text_only)
        furigana_mask = np.array(furigana_mask==0,'B')
        cleaned = cv2.bitwise_not(cleaned)*furigana_mask
        cleaned = cv2.bitwise_not(cleaned)
        text_only = cleaned2segmented(cleaned, average_size)

    (text_like_areas, nontext_like_areas) = filter_text_like_areas(img, segmentation=text_only, average_size=average_size)
    if arg.boolean_value('verbose'):
        print('**********there are ' + str(len(text_like_areas)) + ' text like areas total.')
    text_only = np.zeros(img.shape)
    cc.draw_bounding_boxes(text_only, text_like_areas,color=(255),line_size=-1)

    if arg.boolean_value('debug'):
        text_only = 0.5*text_only + 0.5*img
        #text_rows = 0.5*text_rows+0.5*gray
        #text_colums = 0.5*text_columns+0.5*gray

    segmented_image = np.zeros((h, w, 3), np.uint8)
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
