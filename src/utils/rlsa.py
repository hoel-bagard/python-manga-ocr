"""Run Length Smoothing Algorithm.

Paper can be found there:
https://www.semanticscholar.org/paper/Determination-of-run-length-smoothing-values-for-Papamarkos-Tzortzakis/ab3f8ccf26a0f195eed831af8c167a9c0067457b
https://users.iit.demokritos.gr/~bgat/RLSA_values.pdf

Also found this blog:
http://crblpocr.blogspot.com/2007/06/determination-of-run-length-smoothing.html
http://crblpocr.blogspot.com/2007/06/run-length-smoothing-algorithm-rlsa.html

And this implementation (along with the linked blog post):
https://github.com/Vasistareddy/pythonRLSA
"""
from typing import Optional

import cv2
import numpy as np


def rlsa_horizontal(img: np.ndarray, value: int) -> np.ndarray:
    """Apply the RLS algorithm horizontally on the given image.

    Note: This function can be used to do the operation vertically by simply passing the transpose.

    This function eliminates horizontal white runs whose lengths are smaller than the given value.

    Args:
        img (np.ndarray): The binary image to process.
        value (int): The treshold smoothing value (hsv in the paper).

    Returns:
        The resulting image/mask.
    """
    img = img.copy()
    rows, cols = img.shape
    for row in range(rows):
        count = 0  # Index of the last 0 found
        for col in range(cols):
            if img[row, col] == 0:
                if (col-count) <= value:
                    img[row, count:col] = 0
                count = col
    return img


def rlsa(img: np.ndarray, value_horizontal: int, value_vertical: int, ahsv: Optional[int] = None) -> np.ndarray:
    """Run Length Smoothing Algorithm.

    Args:
        img (np.ndarray): The image to process.
        value_horizontal (int): The horizontal threshold (hsv=300 in the paper)
        value_vertical (int): The vertical threshold (vsv=500 in the paper)

    Returns:
        The resulting image.
    """
    horizontal_rlsa = rlsa_horizontal(img, value_horizontal)
    vertical_rlsa = rlsa_horizontal(img.T, value_vertical).T
    combined_result = cv2.bitwise_and(horizontal_rlsa, vertical_rlsa)
    rlsa_result = rlsa_horizontal(combined_result, ahsv if ahsv else value_horizontal // 10)
    return rlsa_result


if __name__ == "__main__":
    def test():
        from argparse import ArgumentParser
        from src.utils.misc import show_img
        parser = ArgumentParser(description="RLSA testing function. Run with 'python -m src.utils.rlsa <path>'.")
        parser.add_argument("img_path", type=str, help="Path to the test image")
        args = parser.parse_args()

        img_path: str = args.img_path

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        print(f"Processing image with shape width: {width} and height: {height}")
        _, binary_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
        rlsa_result = rlsa(binary_img, 10, 10)  # hsv and vsv are kinda arbitrary here

        show_img(cv2.hconcat([img, rlsa_result]))
    test()
