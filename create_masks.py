import copy
import logging
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from src.text_bboxes import get_text_bboxes
from src.utils.logger import create_logger
from src.utils.misc import clean_print, show_img



def main():
    parser = ArgumentParser(description="OCR to read manga using Tesseract")
    parser.add_argument("img_path", type=Path, help="Path to the image to process.")
    parser.add_argument("--output_folder", "-o", type=Path, default=None, help="Path where to the output folder.")
    parser.add_argument("--display_images", "-d", action="store_true", help="Displays some debug images.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    img_path: Path = args.img_path
    output_path: Path = args.output_folder
    verbose_level: str = args.verbose_level
    display_images: bool = args.display_images
    logger = create_logger("Manga OCR", verbose_level=verbose_level)

    img = cv2.imread(str(img_path), 0)
    result_img = img.copy()
    if display_images and verbose_level == "debug":
        show_img(img, "Input image")

    logger.info("Looking for potential text areas in the image, this might take a while (~20s).")
    text_bounding_boxes = get_text_bboxes(img, logger)
    logger.info(f"Found {len(text_bounding_boxes)} potential text areas. Now doing OCR on them.")
    padding = 30
    for i, (left, top, width, height) in enumerate(text_bounding_boxes, start=1):
        clean_print(f"Processing area {i}/{len(text_bounding_boxes)}", end="\r")
        logger.debug(f"Processing area with width {width} and height {height}    ({i}/{len(text_bounding_boxes)})")
        # The bouding boxing are pretty tight, add a padding aroung it.
        # Tesseract website: "if you OCR just text area without any border, Tesseract could have problems with it."

        padded_crop = cv2.copyMakeBorder(img[top:top+height, left:left+width],
                                         padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)

        detected_text = process_area(padded_crop, logger, display_images)
        if detected_text is not None:
            logger.debug(f"Detected text: {detected_text}")
            # Replace original text by the rendered detected one.
            result_img[top:top+height, left:left+width] = render_text(detected_text, width, height)

        # result_img[top:top+height, left:left+width] = 150

    logger.info("Finished processing the image.")

    if display_images:
        show_img(cv2.hconcat([img, result_img]), "Result")
    if output_path is not None:
        # rel_path = img_path.relative_to(data_path)
        # output_path = output_dir / rel_path.stem
        logger.info(f"Saved result at {output_path}")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), result_img)


if __name__ == "__main__":
    main()
