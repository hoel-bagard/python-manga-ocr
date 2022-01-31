import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.my_types import OCRData


def render_detected_tesseract(img: np.ndarray,
                              ocr_data: OCRData,
                              remove_bg: bool = False,
                              draw_bbox: bool = False) -> np.ndarray:
    """Render the characters detected by Tesseract on the given image.

    Args:
        img (np.ndarray): The image fed to Tesseract.
        ocr_data (OCRData): The predictions from Tesseract
        remove_bg (bool): If True, then removethe parts of the image where some text has been detected to not have
                          the original text and the detected text overlap.
        draw_bbox (bool): If True then draw the bounding box around each block (line) of text.

    Returns:
        The image with the predictions on it.
    """
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    bboxes = list(zip(ocr_data["text"], ocr_data["left"], ocr_data["top"], ocr_data["width"], ocr_data["height"]))
    vertical_margin = 20  # Top / bottom margin when removing the background / text behind the bbox.

    # Clear the background first since the bounding boxes might overlapp (in which case text gets erased)
    if remove_bg:
        for _, left, top, width, height in bboxes:
            draw.rectangle((left, top-vertical_margin, left+width, top+height+vertical_margin), fill=255, width=0)

    for text, left, top, width, height in bboxes:
        # Increase font size until it fills the bounding box
        font_size = 7
        font = ImageFont.truetype("data/NotoSansJP-Regular.otf", font_size)
        text_font_width, text_font_height = font.getsize(text, direction="ttb")
        while text_font_width < 0.90*width and text_font_height < 0.90*height:
            font_size += 1
            font = ImageFont.truetype("data/NotoSansJP-Regular.otf", font_size)
            text_font_width, text_font_height = font.getsize(text, direction="ttb")

        draw.text((left, top), text, fill=1, font=font, direction="ttb", stroke_width=0)
        if draw_bbox:
            draw.rectangle((left, top, left+width, top+height), fill=None, outline=None, width=1)

    # Back to opencv
    img = np.asarray(img)
    return img


def render_text(lines: list[str], width, height) -> np.ndarray:
    """Render the given text on a blank image of the given size.

    Args:
        lines (list): A list of strings. Each string should be a vertical line of text.
        width (int): Desired width.
        height (int): Desired height.

    Returns:
        An image of the given size, with the text rendered on it.
    """
    # img = Image.new('L', (width, height), color=255)
    img = Image.new('L', (width, height), color=150)
    draw = ImageDraw.Draw(img)

    longest_line = lines[np.argmax([len(line) for line in lines])]
    padding = 7  # Space between lines, in pixels
    max_width = width/len(lines) - padding*len(lines)  # width per line, minus padding between lines.

    # Increase font size until it fills the bounding box
    font_size = 7
    font = ImageFont.truetype("data/NotoSansJP-Regular.otf", font_size)
    text_font_width, text_font_height = font.getsize(longest_line, direction="ttb")
    while text_font_width < max_width and text_font_height < 0.95*height:
        font_size += 1
        font = ImageFont.truetype("data/NotoSansJP-Regular.otf", font_size)
        text_font_width, text_font_height = font.getsize(longest_line, direction="ttb")

    offset = width - text_font_width
    for line in lines:
        draw.text((offset, 0), line, fill=1, font=font, direction="ttb", stroke_width=0)
        offset -= (text_font_width + padding)

    # Back to opencv
    img = np.asarray(img)
    return img
