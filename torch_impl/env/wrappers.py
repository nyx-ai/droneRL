import numpy as np
import gym.spaces as spaces
import gym


class BaseGridView(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env, new_step_api=True)

    def _create_base_grid(self):
        grid = np.zeros((self.env.side_size, self.env.side_size, 6), dtype=np.float32)

        for (y, x), drone in self.env.drones.items():
            grid[y, x, 0] = 1
            if drone.packet:
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

        return grid


class GridView(BaseGridView):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.side_size, self.side_size, 6), dtype=float
        )

    def observation(self, _):
        grid = self._create_base_grid()
        return {drone.index: grid.copy() for (_, _), drone in self.env.drones.items()}


class WindowedGridView(BaseGridView):
    def __init__(self, env, radius):
        super().__init__(env)
        self.radius = radius
        assert radius > 0, "Radius should be strictly positive"
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.radius * 2 + 1, self.radius * 2 + 1, 6), dtype=float
        )

    def observation(self, _):
        grid = self._create_base_grid()
        
        padded_size = self.env.side_size + 2 * self.radius
        padded_grid = np.zeros((padded_size, padded_size, 6))
        padded_grid[:, :, 5] = 1
        padded_grid[self.radius:-self.radius, self.radius:-self.radius] = grid

        states = {}
        for (y, x), drone in self.env.drones.items():
            top_left_y = y + self.radius
            top_left_x = x + self.radius
            states[drone.index] = padded_grid[
                top_left_y - self.radius: top_left_y + self.radius + 1,
                top_left_x - self.radius: top_left_x + self.radius + 1,
                :
            ].copy()

        return states
