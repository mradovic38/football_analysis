from dataclasses import dataclass
from typing import Tuple

@dataclass(init=True)
class Club:
    """
    A class to represent a football team.

    Attributes:
        name (str): The name of the club.
        player_jersey_color (Tuple[int, int, int]): The jersey color of the players in RGB format.
        goalkeeper_jersey_color (Tuple[int, int, int]): The jersey color of the goalkeeper in RGB format.
    """
    name: str
    player_jersey_color: Tuple[int, int, int]
    goalkeeper_jersey_color: Tuple[int, int, int]