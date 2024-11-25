import random

from tqdm import tqdm


class DroneEnvironment:
    def __init__(self, env_size, num_drones):
        self.env_size = env_size
        self.drones = {}
        self.available_positions = set((x, y) for x in range(env_size) for y in range(env_size))
        self._populate_environment(num_drones)

    def _populate_environment(self, num_drones):
        positions = random.sample(list(self.available_positions), num_drones)
        for i in range(num_drones):
            position = positions.pop()
            self.drones[position] = f"drone_{i}"
            self.available_positions.remove(position)

    def step(self):
        new_drones = {}
        crashed_drones = []
        newly_occupied_positions = []

        for position, drone_id in list(self.drones.items()):
            del self.drones[position]
            # self.available_positions.add(position)

            move = random.choice([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])
            new_position = (position[0] + move[0], position[1] + move[1])

            if 0 <= new_position[0] < self.env_size and 0 <= new_position[1] < self.env_size:
                if new_position in new_drones:
                    # crashed! into another drone
                    # print(f"{drone_id} crashed into other drone!")
                    crashed_drones.append(drone_id)
                    # print(f"{new_drones[new_position]} was crashed into!")
                    crashed_drones.append(new_drones[new_position])
                    del new_drones[new_position]
                else:
                    # moved successfully
                    # print(f"{drone_id} moved ok to {new_position}")
                    new_drones[new_position] = drone_id
                    # self.available_positions.discard(new_position)
                    # newly_occupied_positions.append(new_position)
            else:
                # crashed! out of env
                # print(f"{drone_id} crashed into wall!")
                crashed_drones.append(drone_id)

        self.drones = new_drones
        # for pos in newly_occupied_positions:
        #     self.available_positions.discard(pos)

        for crashed_drone in crashed_drones:
            self._respawn(crashed_drone)

    # def _respawn(self, drone_id):
    #     new_position = self._select_new_position(self.available_positions)
    #     self.drones[new_position] = drone_id
    #     self.available_positions.remove(new_position)
    # def _select_new_position(self, available_positions):
    #     return random.choice(list(available_positions))

    def _respawn(self, drone_id):
        while True:
            new_position = (
                random.randint(0, self.env_size - 1),
                random.randint(0, self.env_size - 1)
            )
            if new_position not in self.drones:
                self.drones[new_position] = drone_id
                break

    def visualize(self):
        grid = [["." for _ in range(self.env_size)] for _ in range(self.env_size)]
        for position in self.drones:
            x, y = position
            grid[x][y] = "D"
        return "\n".join("".join(row) for row in grid)


if __name__ == '__main__':
    env = DroneEnvironment(env_size=16, num_drones=3)
    pbar = tqdm(range(1_000_000))
    for i in pbar:
        pbar.set_description(f"STEP {i} ({len(env.drones.keys())} drones, {len(env.available_positions)} available pos)")
        # print(env.visualize())
        env.step()
