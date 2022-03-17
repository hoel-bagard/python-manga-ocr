from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.utils.my_types import BBox


class TextRendered:
    def __init__(self, text_folder_path: Path, font_folder_path: Path):
        # Load and parse the text files, keep everything in memory.
        pass

    def render_text(self, img: npt.NDArray[np.uint8], bbox: BBox):
        pass
