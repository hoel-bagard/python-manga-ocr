from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    # First step of the process is to threshold the image using this threshold
    binary_threshold: int = 230

    # The min and max scales are used to filter out components in the get_text_bboxes function.
    min_scale: float = 0.15  # Making this bigger risks making some "." filtered out, be careful.
    max_scale: float = 6.

    # At the end of the text detection based on connected components part,
    # all the bounding boxes whose area is inferior to this threshold are discarded
    # Used to be 5000, but the bubble filtering is pretty good, so it's better to keep a "low" value now.
    min_bbox_area: int = 1000

    # Values used in the RLSA algorithm, used to be based on the average cc size.
    vsv: int = 35
    hsv: int = 35


def get_generation_config() -> GenerationConfig:
    return GenerationConfig()
