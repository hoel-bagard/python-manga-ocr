from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    # First step of the process is to threshold the image using this threshold
    binary_threshold: int = 230

    # At the end of the text detection based on connected components part,
    # all the bounding boxes whose area is inferior to this threshold are discarded
    min_bbox_area: int = 5000

    # Values used in the RLSA algorithm, used to be based on the average cc size.
    vsv: int = 35
    hsv: int = 35


def get_generation_config() -> GenerationConfig:
    return GenerationConfig()
