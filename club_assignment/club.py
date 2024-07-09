from dataclasses import dataclass

@dataclass(init=True)
class Club:
    '''
    A class to represent a team in a football match.
    '''
    name: str
    player_jersey_color: tuple
    goalkeeper_jersey_color: tuple

