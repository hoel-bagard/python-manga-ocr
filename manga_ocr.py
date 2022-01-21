from argparse import ArgumentParser
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import pytesseract

from src.utils.logger import create_logger


class OCRData(TypedDict):
    """Format of the data returned by pytesseract.image_to_data."""

    level: list[int]
    page_num: list[int]
    block_num: list[int]
    line_num: list[int]  # Tesseract tries to split the text into lines. (use it, it works pretty well).
    par_num: list[int]
    word_num: list[int]  # Seems to be the number of a word/character within its line.
    # left, top, width and height can be used to build the bbox around the word.
    left: list[int]
    top: list[int]
    width: list[int]
    height: list[int]
    conf: list[int]  # Confidence level, between 0 and 100.
    text: list[str]  # The detected characters/words.


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


def main():
    parser = ArgumentParser(description="OCR to read manga using Tesseract")
    parser.add_argument("img_path", type=Path, help="Path to the image to process.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    img_path: Path = args.img_path
    verbose_level: str = args.verbose_level

    logger = create_logger("Manga reader", verbose_level=verbose_level)

    img = cv2.imread(str(img_path), 0)

    # height, width = img.shape
    # img = cv2.resize(img, (2*width, 2*height))

    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # padding = 40
    # img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    show_img(img)
    ocr_data: OCRData = pytesseract.image_to_data(img, config="--psm 3", lang="jpn+jpn_vert",
                                                  output_type=pytesseract.Output.DICT)

    print(ocr_data)
    if len((valid_confs := [conf for conf in ocr_data['conf'] if conf != '-1'])) != 0:
        mean_conf = np.mean(valid_confs)
    else:
        mean_conf = 0
    logger.info("Finished processing the frames with Tesseract."
                f" Average confidence for the image: {mean_conf:.2f}")

    # If the average confidence is too low, then the result was probably garbage generated by noise.
    if mean_conf < 10:
        return []

    lines: list[list[tuple[str, int]]] = [[] for _ in range(max(ocr_data["line_num"])+1)]
    for char, conf, line_num in zip(ocr_data["text"], ocr_data['conf'], ocr_data["line_num"]):
        # Skip empty characters
        if char == '':
            continue

        # Remove special characters  (most likely errors)
        # if char in "<>{}[];`@#$%^*_=~\\":
        #     continue

        lines[line_num].append((char, conf))

    lines = [line for line in lines if len(line) != 0]  # Remove empty lines

    logger.info("OCR results:")
    for line in lines:
        logger.info("".join([elt[0] for elt in line]))

    # # Convert the (char/words, conf) tuples to actual lines of text.
    # current_lines = ["".join([elt[0] for elt in line]) for line in ocr_data]

    # logger.info("Finished processing the video.")


if __name__ == "__main__":
    main()
