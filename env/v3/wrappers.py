import numpy as np
import gym.spaces as spaces
import gym


class WindowedGridView(gym.ObservationWrapper):
    def __init__(self, env, radius):
        super().__init__(env)
        self.radius = radius
        assert radius > 0, "Radius should be strictly positive"
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.radius * 2 + 1, self.radius * 2 + 1, 6), dtype=float
        )

    def observation(self, _):
        # Compute full padded grid size
        full_size = self.env.side_size + 2 * self.radius
        padded_grid = np.zeros((full_size, full_size, 6), dtype=np.float32)

        # Mark walls as obstacles initially
        padded_grid[:, :, 5] = 1

        # Reference to the actual grid part excluding padding
        grid = padded_grid[self.radius:-self.radius, self.radius:-self.radius]

        # Mark drones, packets, dropzones, stations, obstacles
        for (y, x), drone in self.env._drones.items():
            grid[y, x, 0] = 1
            if drone.packet is not None:
                grid[y, x, 1] = 1
            grid[y, x, 4] = drone.charge / 100

        # TODO could make this faster by using only two dicts:
        # air and ground, and putting skyscrapers in both.
        for (y, x) in self.env._packets.keys():
            grid[y, x, 1] = 1

        for (y, x) in self.env._dropzones.keys():
            grid[y, x, 2] = 1

        for (y, x) in self.env._stations.keys():
            grid[y, x, 3] = 1

        for (y, x) in self.env._skyscrapers.keys():
            grid[y, x, 5] = 1

        states = {}
        # Extract windowed views for each drone
        for (y, x), drone in self.env._drones.items():
            # Calculate absolute positions with padding offset
            top_left_y = y + self.radius
            top_left_x = x + self.radius
            states[drone.index] = padded_grid[
                                  top_left_y - self.radius: top_left_y + self.radius + 1,
                                  top_left_x - self.radius: top_left_x + self.radius + 1,
                                  :
                                  ]

        return states
