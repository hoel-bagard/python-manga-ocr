from typing import TypedDict


class OCRData(TypedDict):
    """Format of the data returned by pytesseract.image_to_data."""

    level: list[int]  # From 1 to 5, indicates which level the block belongs to (page, block, paragraph, line, word)
    page_num: list[int]
    block_num: list[int]
    line_num: list[int]  # Tesseract tries to split the text into lines. (use it, it works pretty well).
    # par_num: list[int]  # Doesn't seem to exist ?
    word_num: list[int]  # Seems to be the number of a word/character within its line.
    # left, top, width and height can be used to build the bbox around the word.
    left: list[int]
    top: list[int]
    width: list[int]
    height: list[int]
    conf: list[int]  # Confidence level, between 0 and 100.
    text: list[str]  # The detected characters/words.
