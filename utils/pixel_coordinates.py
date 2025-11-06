import torch
from typing import Tuple

#----------------------------
# Convert to Pixel Coordinate
#----------------------------
def to_pixel_coordinates(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    #---Convert normalized coordinates to image coordinates---
    #---From [x_center, y_center, w, h]---
    #---To [x1, y1, x2, y1---

    image_height, image_width = image_size
    x_center, y_center, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = (x_center - w / 2) * image_width
    y1 = (y_center - h / 2) * image_height
    x2 = (x_center + w / 2) * image_width
    y2 = (y_center + h / 2) * image_height

    return torch.stack([x1, y1, x2, y2], dim = -1)