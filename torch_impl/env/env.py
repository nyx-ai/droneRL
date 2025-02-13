import random
import gym.spaces as spaces
import math
from gym import Env


class Drone():
    def __init__(self, index):
        self.index = index
        self.packet = False
        self.charge = 100

    def __repr__(self):
        return f'D{self.index}, packet={self.packet}, charge={self.charge}'


class DeliveryDrones(Env):
    """
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    STAY = 4
    """
    ACTION_TO_DIRECTION = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    NUM_ACTIONS = len(ACTION_TO_DIRECTION)
    DEFAULT_CONFIG = {
        'drone_density': 0.05,
        'n_drones': 3,
        'pickup_reward': 0,
        'delivery_reward': 1,
        'crash_reward': -1,
        'charge_reward': -0.1,
        'discharge': 10,
        'charge': 20,
        'packets_factor': 3,
        'dropzones_factor': 2,
        'stations_factor': 2,
        'skyscrapers_factor': 3,
        'rgb_render_rescale': 1.0,
    }

    metadata = {
        'render.modes': ['ansi'],
    }

    @property
    def drones_list(self):
        return list(self.drones.values())

    def __init__(self, env_params={}):
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.env_params = self.DEFAULT_CONFIG
        self.env_params.update(env_params)
        self.reset()

    def spawn_objects(self, available_pos, num_obj):
        if len(available_pos) < num_obj:
            raise ValueError(f"Not enough positions ({len(available_pos)}) to spawn {num_obj} objects")
        positions_dict = {}
        random.shuffle(available_pos)
        for _ in range(num_obj):
            position = available_pos.pop()
            positions_dict[position] = True
        return positions_dict, available_pos

    def reset(self, seed=0):
        self.drones = {}
        self.stations = {}
        self.dropzones = {}
        self.packets = {}
        self.skyscrapers = {}
        self.n_drones = self.env_params['n_drones']
        self.side_size = int(math.ceil(math.sqrt(self.env_params['n_drones'] / self.env_params['drone_density'])))
        self.shape = (self.side_size, self.side_size)

        # Create elements of the grid
        num_skyscrapers = self.env_params['skyscrapers_factor'] * self.env_params['n_drones']
        num_packets = self.env_params['packets_factor'] * self.env_params['n_drones']
        num_dropzone = self.env_params['dropzones_factor'] * self.env_params['n_drones']
        num_stations = self.env_params['stations_factor'] * self.env_params['n_drones']
        available_positions = [(x, y) for x in range(self.side_size) for y in range(self.side_size)]
        self.skyscrapers, available_positions = self.spawn_objects(available_positions, num_skyscrapers)

        # Add the drones, which don't remove their positions from available_positions
        # as they can spawn on packets, dropzones or stations
        for i, p in enumerate(random.sample(available_positions, self.n_drones)):
            self.drones[p] = Drone(i)

        self.packets, available_positions = self.spawn_objects(
            available_positions, num_packets)
        self.dropzones, available_positions = self.spawn_objects(
            available_positions, num_dropzone)
        self.stations, available_positions = self.spawn_objects(
            available_positions, num_stations)

        # Check if some packets are immediately picked
        self._pick_packets_after_respawn()

        return self.get_state(), None

    def get_state(self):
        return {
            'drones': self.drones,
            'stations': self.stations,
            'dropzones': self.dropzones,
            'packets': self.packets,
            'skyscrapers': self.skyscrapers,
        }

    def step(self, actions):
        info = {}
        rewards = {index: 0 for index in actions.keys()}
        dones = {index: False for index in actions.keys()}

        new_drones = {}
        crashed_drones = []
        crashed_drone_locations = []
        nb_dropzones_to_respawn = 0
        nb_packets_to_respawn = 0

        # Move all drones to their new location
        for position, drone in self.drones.items():
            move = self.ACTION_TO_DIRECTION[actions[drone.index]]
            new_position = (position[0] + move[0], position[1] + move[1])

            if 0 <= new_position[0] < self.side_size and 0 <= new_position[1] < self.side_size:
                if new_position in new_drones:
                    # crashed into another drone
                    #print(f"COLLISION between drones {drone} & {new_drones[new_position]}!!")
                    crashed_drones.append(drone)
                    crashed_drone_locations.append(new_position)
                else:
                    # moved successfully
                    new_drones[new_position] = drone
            else:
                # crashed out of env bounds
                #print(f"OUT OF BOUNDS from drone {drone}!!")
                crashed_drones.append(drone)

        # Handle drones that didn't crash yet
        for position, drone in new_drones.items():
            if drone not in crashed_drones:
                # charging/discharging
                if position in self.stations:
                    drone.charge = min(100, drone.charge + self.env_params['charge'])
                    rewards[drone.index] = self.env_params['charge_reward']
                else:
                    drone.charge -= self.env_params['discharge']
                    if drone.charge <= 0:
                        #print(f"DEAD BATTERY from drone {drone}!!")
                        crashed_drone_locations.append(position)

                # pickup/delivery
                if position in self.packets and not drone.packet:
                    #print(f"PICKUP from drone {drone}!!")
                    rewards[drone.index] = self.env_params['pickup_reward']
                    drone.packet = True
                    del self.packets[position]
                elif position in self.dropzones and drone.packet:
                    #print(f"DELIVERY from drone {drone}!!")
                    rewards[drone.index] = self.env_params['delivery_reward']
                    drone.packet = False
                    del self.dropzones[position]
                    nb_dropzones_to_respawn += 1
                    nb_packets_to_respawn += 1

                # skyscrapers
                if position in self.skyscrapers:
                    #print(f"CRASHED by entering a skyscraper: {drone}!!")
                    crashed_drone_locations.append(position)

        # Clean up locations where crashes occurred
        # we need to do that AFTER we've iterated over all drones,
        # as otherwise we could miss some collisions
        for crashed_drone_location in crashed_drone_locations:
            if crashed_drone_location in new_drones:
                #print(f"CRASHED by entering a crashed location: {new_drones[crashed_drone_location]}!!")
                crashed_drones.append(new_drones[crashed_drone_location])
                del new_drones[crashed_drone_location]

        self.drones = new_drones

        # Respawn crashed drones
        for crashed_drone in crashed_drones:
            crashed_drone.charge = 100
            if crashed_drone.packet:
                nb_packets_to_respawn += 1
                crashed_drone.packet = False
            rewards[crashed_drone.index] = self.env_params['crash_reward']
            dones[crashed_drone.index] = True
            respawn_position = self._find_respawn_position(self.drones | self.skyscrapers)
            self.drones[respawn_position] = crashed_drone
            #print(f"Respawned crashed drone {crashed_drone} at {respawn_position}")

        # Respawn used packets and dropzones
        ground_mask = {}
        ground_mask.update(self.skyscrapers)
        ground_mask.update(self.packets)
        ground_mask.update(self.dropzones)
        ground_mask.update(self.stations)
        for _ in range(nb_packets_to_respawn):
            position = self._find_respawn_position(ground_mask)
            self.packets[position] = True
            ground_mask[position] = True
        for _ in range(nb_dropzones_to_respawn):
            position = self._find_respawn_position(ground_mask)
            self.dropzones[position] = True
            ground_mask[position] = True

        # check if some packets are immediately picked
        self._pick_packets_after_respawn()

        return self.get_state(), rewards, dones, None, info

    def _pick_packets_after_respawn(self):
        for drone_pos, drone in self.drones.items():
            if drone.packet is False and drone_pos in self.packets:
                # we don't give pickup_reward in this case
                # as the drone didn't do anything to deserve it
                #print(f"SPAWN PICKUP from {drone}!")
                drone.packet = True
                del self.packets[drone_pos]

    def _find_respawn_position(self, mask={}):
        while True:
            p = (
                random.randint(0, self.side_size - 1),
                random.randint(0, self.side_size - 1)
            )
            if p not in mask:
                return p

    def render(self, mode='ansi'):
        return self.__str__()

    def __str__(self):
        lines = ["_" * self.shape[0] * 2]
        for y in range(self.shape[0]):
            line_str = ''
            for x in range(self.shape[1]):
                position = (y, x)
                if position in self.drones:
                    drone = self.drones[position]
                    tile_char = f'{drone.index}'
                elif position in self.packets:
                    tile_char = 'x'
                elif position in self.dropzones:
                    tile_char = 'D'
                elif position in self.stations:
                    tile_char = '@'
                elif position in self.skyscrapers:
                    tile_char = '#'
                else:
                    tile_char = '.'
                line_str += tile_char.ljust(2)
            lines.append(line_str)
        lines.append("_" * self.shape[0] * 2)
        return '\n'.join(lines)
