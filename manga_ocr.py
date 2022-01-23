import copy
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from src.my_types import OCRData
from src.render_ocr import render_detected
from src.utils.logger import create_logger


def show_img(img: np.ndarray, window_name: str = "Image"):
    """Displays an image until the user presses the "q" key.

    Args:
        img: The image that is to be displayed.
        window_name (str): The name of the window in which the image will be displayed.
    """
    while True:
        # Make the image full screen if it's above a given size (assume the screen isn't too small^^)
        if any(img.shape[:2] > np.asarray([1080, 1440])):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def clean_ocr_results(ocr_data: OCRData) -> OCRData:
    """Remove unwanted characters from the Tesseract output.

    Has the side effect of removing empty lines.

    Args:
        ocr_data: The raw output from the pytesseract.image_to_data function.

    Returns:
        Same type of data, but without the noise.
    """
    clean_ocr_data: OCRData = copy.deepcopy(ocr_data)  # Not necessary but feels safer.
    for i in range(len(ocr_data["text"])-1, -1, -1):
        char = ocr_data["text"][i]
        # Remove empty and special characters  (most likely errors)
        if char == '' or char in "<>{}[];`@#$%^*_=~\\":
            for key in ocr_data.keys():
                clean_ocr_data[key].pop(i)
    return clean_ocr_data


def ocr_char_to_block(ocr_data: OCRData) -> OCRData:
    """Reformats the ocr data to have entries correspond to blocks instead of characters.

    Note: The width and height values can get quite wrong since errors accumulate.

    Args:
        ocr_data: Standard OCR data where most entries correspond to one or two characters.

    Returns:
        Same type of data, but each entry corresponds to a block of characters.
    """
    nb_blocks = max(ocr_data["block_num"])
    # Not sure if there is a better way to instanciate a TypedDict.
    block_ocr_data: OCRData = {
        "level": [4 for _ in range(nb_blocks)],
        "page_num": [1 for _ in range(nb_blocks)],
        "block_num": [i for i in range(1, nb_blocks+1)],
        "line_num": [i for i in range(1, nb_blocks+1)],
        "word_num": [1 for _ in range(nb_blocks)],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "conf": [],
        "text": [],
    }
    for i in range(1, nb_blocks+1):
        block_ocr_data["left"].append(min([ocr_data["left"][j]
                                           for j in range(len(ocr_data["text"])) if ocr_data["block_num"][j] == i]))
        block_ocr_data["top"].append(min([ocr_data["top"][j]
                                          for j in range(len(ocr_data["text"])) if ocr_data["block_num"][j] == i]))
        block_ocr_data["width"].append(max([ocr_data["width"][j]
                                            for j in range(len(ocr_data["text"])) if ocr_data["block_num"][j] == i]))
        block_ocr_data["height"].append(sum([ocr_data["height"][j]
                                             for j in range(len(ocr_data["text"])) if ocr_data["block_num"][j] == i]))
        block_ocr_data["conf"].append(np.mean([ocr_data["conf"][j]
                                               for j in range(len(ocr_data["text"])) if ocr_data["block_num"][j] == i]))
        block_ocr_data["text"].append(''.join([ocr_data["text"][j]
                                               for j in range(len(ocr_data["text"])) if ocr_data["block_num"][j] == i]))
    return block_ocr_data


def main():
    parser = ArgumentParser(description="OCR to read manga using Tesseract")
    parser.add_argument("img_path", type=Path, help="Path to the image to process.")
    parser.add_argument("--display_images", "-d", action="store_true", help="Displays some debug images.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    img_path: Path = args.img_path
    verbose_level: str = args.verbose_level
    display_images: bool = args.display_images
    logger = create_logger("Manga reader", verbose_level=verbose_level)

    img = cv2.imread(str(img_path), 0)

    # height, width = img.shape
    # img = cv2.resize(img, (2*width, 2*height))

    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # padding = 40
    # img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    if display_images:
        show_img(img, "Input image")

    # Notes on the (py)Tesseract options:
    # For the config --psm, use either 5 or 12:
    #     5  Assume a single uniform block of vertically aligned text.
    #     12 Sparse text with OSD.
    # From my limited testing, 12 is more accurate, but does not separate lines (5 does).
    # For the lang, use either "jpn+jpn_vert" or "jpn_vert".
    ocr_data: OCRData = pytesseract.image_to_data(img, config="--psm 12", lang="jpn_vert",
                                                  output_type=pytesseract.Output.DICT)
    logger.debug(f"Frame processed by Tesseract. Raw output:\n{ocr_data}")

    if len((valid_confs := [conf for conf in ocr_data['conf'] if conf != '-1'])) != 0:
        mean_conf = np.mean(valid_confs)
    else:
        mean_conf = 0
    # If the average confidence is too low, then the result was probably garbage generated by noise.
    if mean_conf < 10:
        logger.info("Did not find any text on the image.")
        exit()
    logger.info("Finished processing the frames with Tesseract."
                f" Average confidence for the image: {mean_conf:.2f}")

    ocr_data = clean_ocr_results(ocr_data)
    logger.debug(f"Cleaned OCR data:\n{ocr_data}")

    ocr_data = ocr_char_to_block(ocr_data)

    if display_images:
        result_img = render_detected(img, ocr_data)
        show_img(result_img)

    all_text = "\n\t".join(ocr_data["text"])
    logger.info("OCR results:\n\t" + all_text)

    logger.info("Finished processing the image.")


if __name__ == "__main__":
    main()
