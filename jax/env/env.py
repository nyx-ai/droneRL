from typing import Optional, Dict, Tuple, Union
import jax
import jax.numpy as jnp
import jax.random
from tqdm import trange
from flax.struct import dataclass


@dataclass
class EnvState:
    pass


@dataclass
class EnvParams:
    pass


class Env:
    "Environment base class for JAX"
    def reset(self, key: jnp.array):
        raise NotImplementedError

    def step(
            self,
            key: jnp.array,
            state: EnvState,
            action: jnp.array):
        raise NotImplementedError


@dataclass
class DroneEnvParams(EnvParams):
    drone_density: float = 0.05
    n_drones: int = 3
    pickup_reward: float = 0.0
    delivery_reward: float = 1.0
    crash_reward: float = -1.0
    charge_reward: float = -0.1
    discharge: int = 10
    charge: int = 20
    packets_factor: int = 3
    dropzones_factor: int = 2
    stations_factor: int = 2
    skyscrapers_factor: int = 3
    rgb_render_rescale: float = 1.0


@dataclass
class DroneEnvState(EnvState):
    ground: jnp.ndarray
    air: jnp.ndarray
    carrying_package: jnp.ndarray
    charge: jnp.ndarray


NUM_ACTIONS = 5
ACTION_LEFT = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_STAY = 4

OBJ_SKYSCRAPER = 2
OBJ_STATION = 3
OBJ_DROPZONE = 4
OBJ_PACKET = 5


class DeliveryDrones(Env):
    def __init__(self, env_params: None) -> None:
        if env_params is None:
            self.params = DroneEnvParams()
        else:
            self.params = env_params

    def spawn(
            self,
            key: jnp.ndarray,
            grid: jnp.ndarray,
            num_elements: int,
            fill_value: Union[int, jnp.ndarray],
            exclude_positions: Optional[jnp.ndarray] = None
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        valid_mask = grid == 0
        if exclude_positions is not None:
            exclude_mask = jnp.ones_like(grid, dtype=jnp.bool)
            exclude_mask = exclude_mask.at[exclude_positions].set(False)
            valid_mask &= exclude_mask

        all_indices = jnp.arange(grid.size)
        H, W = grid.shape
        selected_indices = jax.random.choice(
                key,
                all_indices,
                shape=(num_elements,),
                replace=False,
                p=valid_mask.ravel())
        spawn_y = selected_indices // W
        spawn_x = selected_indices % W

        spawn_mask = jnp.zeros_like(grid, dtype=jnp.bool)
        spawn_mask = spawn_mask.at[spawn_y, spawn_x].set(True)
        spawn_values = jnp.zeros_like(grid, dtype=jnp.int32)
        fill_values_flat = jnp.asarray(fill_value).reshape(-1)
        spawn_values = spawn_values.at[spawn_y, spawn_x].set(fill_values_flat[:num_elements])
        new_grid = jnp.where(spawn_mask, spawn_values, grid)
        return new_grid, (spawn_y, spawn_x)

    def reset(self, key: jnp.array):
        # build grids
        grid_size = int(jnp.ceil(jnp.sqrt(self.params.n_drones / self.params.drone_density)).item())
        air = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)  # TODO check whether uint8 grids could be used instead
        ground = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        # place objects
        num_packets = self.params.packets_factor * self.params.n_drones
        num_dropzones = self.params.dropzones_factor * self.params.n_drones
        num_stations = self.params.stations_factor * self.params.n_drones
        num_skyscrapers = self.params.skyscrapers_factor * self.params.n_drones
        key, spawn_key = jax.random.split(key)
        ground, _ = self.spawn(spawn_key, ground, num_packets, OBJ_PACKET)
        key, spawn_key = jax.random.split(key)
        ground, _ = self.spawn(spawn_key, ground, num_dropzones, OBJ_DROPZONE)
        key, spawn_key = jax.random.split(key)
        ground, _ = self.spawn(spawn_key, ground, num_stations, OBJ_STATION)
        key, spawn_key = jax.random.split(key)
        ground, skyscrapers_pos = self.spawn(spawn_key, ground, num_skyscrapers, OBJ_SKYSCRAPER)
        key, spawn_key = jax.random.split(key)
        air, _ = self.spawn(spawn_key, air, self.params.n_drones, jnp.arange(1, self.params.n_drones + 1), exclude_positions=skyscrapers_pos)
        state = DroneEnvState(
                air=air,
                ground=ground,
                carrying_package=jnp.zeros(self.params.n_drones, dtype=jnp.bool),
                charge=100 * jnp.ones(self.params.n_drones, dtype=jnp.float32)
                )
        return state

    def step(
            self,
            key: jnp.array,
            state: DroneEnvState,
            actions: jnp.array,
             ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # get current drone positions
        drone_y, drone_x = jnp.where(state.air > 0, size=self.params.n_drones)
        drone_ids = state.air[drone_y, drone_x]
        sort_idx = jnp.argsort(drone_ids)
        drone_y = drone_y[sort_idx]
        drone_x = drone_x[sort_idx]
        drone_ids = drone_ids[sort_idx]

        # compute new position of drones
        dy = jnp.where(actions == ACTION_UP, -1, jnp.where(actions == ACTION_DOWN, 1, 0))
        dx = jnp.where(actions == ACTION_LEFT, -1, jnp.where(actions == ACTION_RIGHT, 1, 0))
        new_y = drone_y + dy
        new_x = drone_x + dx

        # check off board
        grid_size = state.air.shape[0]
        off_board = (new_y < 0) | (new_y >= grid_size) | (new_x < 0) | (new_x >= grid_size)

        # check skyscrapers
        on_board_y = jnp.clip(new_y, 0, grid_size - 1)
        on_board_x = jnp.clip(new_x, 0, grid_size - 1)
        collided_skyscrapers = (state.ground[on_board_y, on_board_x] == OBJ_SKYSCRAPER) & ~off_board

        # collisions between drones
        new_pos = jnp.stack([new_y, new_x], axis=-1)
        unique_positions, inverse_indices, counts = jnp.unique(
                new_pos,
                axis=0,
                return_inverse=True,
                return_counts=True,
                size=self.params.n_drones)
        collisions = counts[inverse_indices] > 1

        # check whether we can pick up a package
        dones = off_board | collided_skyscrapers | collisions
        survivors = ~dones
        picked_up = (state.ground[new_y, new_x] == OBJ_PACKET) & survivors & ~state.carrying_package
        carrying_package = state.carrying_package | picked_up
        mask = jnp.zeros_like(state.ground, dtype=jnp.bool)
        mask = mask.at[new_y, new_x].set(picked_up)
        ground = jnp.where(mask, 0, state.ground)

        # check whether we can deliver a package
        dropoff_positions = (state.ground[new_y, new_x] == OBJ_DROPZONE) & survivors
        delivered = dropoff_positions & carrying_package
        carrying_package = carrying_package & ~delivered

        # respawn delivered packages
        num_delivered = jnp.sum(delivered)
        key, spawn_key = jax.random.split(key)
        ground, _ = self.spawn(
                spawn_key,
                grid=ground,
                num_elements=num_delivered,
                fill_value=OBJ_PACKET)

        # handle charge
        is_charging = (state.ground[new_y, new_x] == OBJ_STATION) & survivors & (state.charge < 100)
        is_discharging = (state.ground[new_y, new_x] != OBJ_STATION) & survivors
        charge = (state.charge + is_charging * self.params.charge).clip(0, 100)
        charge = (charge - is_discharging * self.params.discharge).clip(0, 100)
        out_of_charge = charge == 0
        dones = dones | out_of_charge

        # compute rewards
        rewards = jnp.zeros(self.params.n_drones, dtype=jnp.float32)
        rewards += self.params.crash_reward * dones
        rewards += self.params.pickup_reward * picked_up
        rewards += self.params.delivery_reward * delivered
        rewards += self.params.charge_reward * is_charging

        # air respawns
        num_respawn = jnp.sum(dones)
        key, spawn_key = jax.random.split(key)
        skyscrapers_pos = jnp.where(state.ground == OBJ_SKYSCRAPER)  # TODO: could be cached
        air = state.air.at[drone_y[dones], drone_x[dones]].set(0)
        air, newly_spawned_idx = self.spawn(
                spawn_key,
                air,
                num_respawn,
                drone_ids[dones],
                exclude_positions=skyscrapers_pos)

        # potentially pick up newly spawned packages
        new_packages_found = state.ground[newly_spawned_idx] == OBJ_PACKET
        carrying_package = carrying_package.at[dones].set(new_packages_found)
        ground = ground.at[newly_spawned_idx].set(0)

        # actually move suriving drones
        air = air.at[drone_y[survivors], drone_x[survivors]].set(0)
        air = air.at[new_y[survivors], new_x[survivors]].set(drone_ids[survivors])

        # update state
        state = state.replace(
                air=air,
                ground=ground,
                charge=charge,
                carrying_package=carrying_package)

        return state, rewards, dones

    def format_action(self, *actions):
        return [['←', '↓', '→', '↑', 'X'][i] for i in actions]


if __name__ == "__main__":
    params = DroneEnvParams(n_drones=1000)
    env = DeliveryDrones(params)
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    # step_jit = jax.jit(env.step)
    step_jit = env.step

    for i in trange(1000):
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (env.params.n_drones,), 0, NUM_ACTIONS, dtype=jnp.int32)
        state, rewards, dones = step_jit(rng, state, actions)
        # __import__('pdb').set_trace()
        # print('actions', [env.format_action(*list(actions))])
        # print('dones', dones)
        # print('charge', state.charge)
        # print('rewards', rewards)
        # print('packets', state.carrying_package)
        # print('after\n', state.air)
