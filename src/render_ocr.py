import numpy as np
from PIL import Image, ImageDraw

from src.my_types import OCRData


def render_detected(img: np.ndarray, ocr_data: OCRData):
    """Render the characters detected by Tesseract on the given image.

    Args:
        img (np.ndarray): The image fed to Tesseract.
        ocr_data (OCRData): The predictions from Tesseract

    Returns:
        The image with the predictions on it.
    """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    # left: list[int]
    # top: list[int]
    # width: list[int]
    # height: list[int]
    # conf: list[int]  # Confidence level, between 0 and 100.
    # text: list[str]  # The detected characters/words.
    useful_values = zip(ocr_data["text"], ocr_data["left"], ocr_data["top"], ocr_data["width"], ocr_data["height"])
    for char, left, top, width, height in useful_values:
        # TODO: clean the bbox.
        draw.text((left, top), char, fill=1)#, font=font, stroke_width=0)

    # Back to opencv
    img = np.asarray(img)
    return img
