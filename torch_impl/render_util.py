from enum import IntEnum
import numpy as np


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


def convert_for_rendering(env):
    side_size = env.side_size
    ground = np.full((side_size, side_size), None)
    for (y, x) in env.dropzones.keys():
        ground[y, x] = Object.DROPZONE
    for (y, x) in env.stations.keys():
        ground[y, x] = Object.STATION
    for (y, x) in env.skyscrapers.keys():
        ground[y, x] = Object.SKYSCRAPER
    for (y, x) in env.packets.keys():
        ground[y, x] = Object.PACKET

    air = np.full((side_size, side_size), None)

    carrying_package = []
    charge = []
    # We sort the drones by index to ensure the order is consistent
    for pos, drone in sorted(env.drones.items(), key=lambda x: x[1].index):
        y, x = pos
        air[y, x] = drone.index
        carrying_package.append(1 if drone.packet else 0)
        charge.append(drone.charge)
    return ground, air, carrying_package, charge
