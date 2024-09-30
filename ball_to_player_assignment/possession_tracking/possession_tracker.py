from club_assignment import Club

from typing import Dict, List

class PossessionTracker:
    """Tracking the ball possession of each club"""

    def __init__(self, club1: Club, club2: Club) -> None:
        """
        Initializes the PossessionTracker with club names and possession statistics.

        Args:
            club1 (Club): The first club object
            club2 (Club): The second club object
        """
        self.possession_dict: Dict[str, int] = {-1: 0, club1.name: 0, club2.name: 0}
        self.club1_name: str = club1.name
        self.club2_name: str = club2.name
        self.possession: List[Dict[int, float]] = []  # List to track possession percentages over time
        self.sum: int = 0  # Total number of possession instances

    def add_possession(self, club_name: str) -> None:
        """
        Records possession for a specific club and updates possession statistics.

        Args:
            club_name (str): The name of the club that currently has possession.
        """
        self.possession_dict[club_name] += 1
        self.sum += 1
        self.possession.append({
            -1: self.possession_dict[-1] / self.sum, 
            0: self.possession_dict[self.club1_name] / self.sum, 
            1: self.possession_dict[self.club2_name] / self.sum
        })