from enum import IntEnum

class Action(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    STAY = 4

    @classmethod
    def num_actions(cls) -> int:
        return len(cls)


class Object(IntEnum):
    SKYSCRAPER = 2
    STATION = 3
    DROPZONE = 4
    PACKET = 5
