import numpy as np
import gym.spaces as spaces
import gym


class GridView(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env, new_step_api=True)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.side_size, self.side_size, 6), dtype=float
        )

    def observation(self, _):
        grid = np.zeros((self.env.side_size, self.env.side_size, 6), dtype=np.float32)
        print(grid.shape)

        for (y, x), drone in self.env.drones.items():
            grid[y, x, 0] = 1
            if drone.packet is not None:
                grid[y, x, 1] = 1
            grid[y, x, 4] = drone.charge / 100

        for (y, x) in self.env.packets.keys():
            grid[y, x, 1] = 1

        for (y, x) in self.env.dropzones.keys():
            grid[y, x, 2] = 1

        for (y, x) in self.env.stations.keys():
            grid[y, x, 3] = 1

        for (y, x) in self.env.skyscrapers.keys():
            grid[y, x, 5] = 1

        states = {}
        for (y, x), drone in self.env.drones.items():
            states[drone.index] = grid.copy()
        return states


class WindowedGridView(gym.ObservationWrapper):
    def __init__(self, env, radius):
        super().__init__(env, new_step_api=True)
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
        for (y, x), drone in self.env.drones.items():
            grid[y, x, 0] = 1
            if drone.packet is not None:
                grid[y, x, 1] = 1
            grid[y, x, 4] = drone.charge / 100

        for (y, x) in self.env.packets.keys():
            grid[y, x, 1] = 1

        for (y, x) in self.env.dropzones.keys():
            grid[y, x, 2] = 1

        for (y, x) in self.env.stations.keys():
            grid[y, x, 3] = 1

        for (y, x) in self.env.skyscrapers.keys():
            grid[y, x, 5] = 1

        states = {}
        # Extract windowed views for each drone
        for (y, x), drone in self.env.drones.items():
            # Calculate absolute positions with padding offset
            top_left_y = y + self.radius
            top_left_x = x + self.radius
            states[drone.index] = padded_grid[
                top_left_y - self.radius: top_left_y + self.radius + 1,
                top_left_x - self.radius: top_left_x + self.radius + 1,
                :
            ].copy()

        return states
