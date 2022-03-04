from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    # First step of the process is to threshold the image using this threshold
    binary_threshold: int = 230

    # At the end of the text detection based on connected components part,
    # all the bounding boxes whose area is inferior to this threshold are discarded
    min_bbox_area: int = 5000


def get_generation_config() -> GenerationConfig:
    return GenerationConfig()
