"""
Some classes for typing
"""

from typing import TypedDict
from typing import Optional

class ReturnDetectionDict(TypedDict):
    orig_image_name : str
    orig_image_fullpath: str
    x_max: Optional[int]
    x_min: Optional[int]
    y_max: Optional[int]
    y_min: Optional[int]
    votings: list
    source: Optional[str]
