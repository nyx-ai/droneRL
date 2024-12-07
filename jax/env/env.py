from typing import Optional, Dict, Tuple, Union
import jax
import jax.numpy as jnp
import jax.random
from tqdm import trange
from flax.struct import dataclass
from timeit import default_timer as timer


@dataclass
class EnvState:
    pass


@dataclass
class EnvParams:
    pass


class Env:
    "Environment base class for JAX"
    def reset(self, key: jax.Array):
        raise NotImplementedError

    def step(
            self,
            key: jax.Array,
            state: EnvState,
            action: jax.Array):
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
    def __init__(self) -> None:
        pass

    def spawn(
            self,
            key: jnp.ndarray,
            grid: jnp.ndarray,
            fill_values: jnp.ndarray,
            exclude_mask: Optional[jnp.ndarray] = None,
            ) -> jnp.ndarray:
        def _place(carry, val):
            grid, key = carry
            key, place_key = jax.random.split(key)
            probs = (grid == 0).astype(float)
            if exclude_mask is not None:
                probs *= (~exclude_mask).astype(float)
            # TODO: Handle case where jnp.sum(probs) is zero, i.e. no positions are available for spawning!
            probs = probs / jnp.sum(probs)
            flat_probs = probs.ravel()
            pos = jax.random.choice(place_key, grid.size, p=flat_probs)
            pos_y, pos_x = jnp.unravel_index(pos, grid.shape)
            grid = grid.at[pos_y, pos_x].set(val)
            return (grid, key), None
        (grid, _), _ = jax.lax.scan(_place, (grid, key), fill_values)
        return grid


    def reset(self, key: jax.Array, params: DroneEnvParams, grid_size: int):
        # build grids
        air = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)  # TODO check whether uint8 grids could be used instead
        ground = jnp.zeros((grid_size, grid_size), dtype=jnp.int8)
        # place objects
        num_packets = params.packets_factor * params.n_drones
        num_dropzones = params.dropzones_factor * params.n_drones
        num_stations = params.stations_factor * params.n_drones
        num_skyscrapers = params.skyscrapers_factor * params.n_drones
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_packets, dtype=jnp.int8) * OBJ_PACKET)
        key, spawn_key = jax.random.split(key)
        ground= self.spawn(spawn_key, ground, jnp.ones(num_dropzones, dtype=jnp.int8) * OBJ_DROPZONE)
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_stations, dtype=jnp.int8) * OBJ_STATION)
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_skyscrapers, dtype=jnp.int8) * OBJ_SKYSCRAPER)
        key, spawn_key = jax.random.split(key)
        air = self.spawn(spawn_key, air, jnp.arange(1, params.n_drones + 1, dtype=jnp.int32), exclude_mask=(ground == OBJ_SKYSCRAPER))

        # check if drones picked up package
        package_mask = (ground == OBJ_PACKET)
        carrying_package = jnp.array([jnp.any((air == player) & package_mask) for player in range(1, params.n_drones + 1)])
        state = DroneEnvState(
                air=air,
                ground=ground,
                carrying_package=carrying_package,
                charge=100 * jnp.ones(params.n_drones, dtype=jnp.float32)
                )
        return state

    def step(
            self,
            key: jax.Array,
            state: DroneEnvState,
            actions: jax.Array,
            params: DroneEnvParams,
             ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # get current drone positions
        drone_y, drone_x = jnp.where(state.air > 0, size=params.n_drones)
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
                size=params.n_drones)
        collisions = counts[inverse_indices] > 1

        # compute who survived
        dones = off_board | collided_skyscrapers | collisions
        survivors = ~dones

        # check whether we can pick up a package
        carrying_package = state.carrying_package & survivors  # only survivors keep their packages
        picked_up = (state.ground[new_y, new_x] == OBJ_PACKET) & survivors & ~carrying_package  # surivors may pick up packages at new locations
        carrying_package |= picked_up
        mask = jnp.zeros_like(state.ground, dtype=jnp.bool)
        mask = mask.at[new_y, new_x].set(picked_up)
        ground = jnp.where(mask, 0, state.ground)

        # check whether we can deliver a package
        dropoff_positions = (state.ground[new_y, new_x] == OBJ_DROPZONE) & survivors
        delivered = dropoff_positions & carrying_package
        carrying_package = carrying_package & ~delivered

        # respawn delivered packages
        key, spawn_key = jax.random.split(key)
        num_packets = params.packets_factor * params.n_drones
        delivered = ~delivered
        fill_values = jnp.zeros(num_packets, dtype=jnp.int8)
        # trick: if no deliveries took place we spawn a "0", else OBJ_PACKET
        fill_values = fill_values.at[:len(delivered)].set(delivered * OBJ_PACKET)
        ground = self.spawn(spawn_key, ground, fill_values)

        # handle charge
        is_charging = (state.ground[new_y, new_x] == OBJ_STATION) & survivors & (state.charge < 100)
        is_discharging = (state.ground[new_y, new_x] != OBJ_STATION) & survivors
        charge = (state.charge + is_charging * params.charge).clip(0, 100)
        charge = (charge - is_discharging * params.discharge).clip(0, 100)
        out_of_charge = charge == 0
        dones = dones | out_of_charge

        # compute rewards
        rewards = jnp.zeros(params.n_drones, dtype=jnp.float32)
        rewards += params.crash_reward * dones
        rewards += params.pickup_reward * picked_up
        rewards += params.delivery_reward * delivered
        rewards += params.charge_reward * is_charging

        # air respawns
        key, spawn_key = jax.random.split(key)
        alive_player_ids = jnp.arange(1, params.n_drones + 1, dtype=jnp.int32) * ~dones  # only alive player IDs are set
        air = state.air.at[drone_y, drone_x].set(alive_player_ids)
        dead_player_ids = jnp.arange(1, params.n_drones + 1, dtype=jnp.int32) * dones  # only dead player IDs are set
        air = self.spawn(
                spawn_key,
                air,
                dead_player_ids,
                exclude_mask=(state.ground == OBJ_SKYSCRAPER))

        # potentially pick up newly spawned packages
        package_mask = (ground == OBJ_PACKET)
        can_pick_up = jnp.array([jnp.any((air == player) & package_mask) for player in range(1, params.n_drones + 1)], dtype=jnp.bool)
        picked_up = can_pick_up & dones  # all the newly spawned players that can now pick up a package
        carrying_package |= picked_up

        # compute new positions of drones after respawns
        drone_y, drone_x = jnp.where(air > 0, size=params.n_drones)
        drone_ids = air[drone_y, drone_x]
        sort_idx = jnp.argsort(drone_ids)
        drone_y = drone_y[sort_idx]
        drone_x = drone_x[sort_idx]

        # remove packages that were picked up by respawned drones
        mask = jnp.zeros_like(state.ground, dtype=jnp.bool)
        mask = mask.at[drone_y, drone_x].set(picked_up)
        ground = jnp.where(mask, 0, state.ground)

        # actually move suriving drones
        mask = jnp.zeros_like(air, dtype=jnp.bool)
        mask = mask.at[drone_y, drone_x].set(survivors)
        air = jnp.where(mask, 0, air)
        air = air.at[new_y, new_x].set(jnp.arange(1, params.n_drones + 1) * survivors)

        # update state
        state = state.replace(
                air=air,
                ground=ground,
                charge=charge,
                carrying_package=carrying_package)

        return state, rewards, dones


    def run_steps(
            self,
            key: jax.Array,
            state: DroneEnvState,
            params: DroneEnvParams,
            num_steps: int,
            ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        def loop_body(i, carry):
            rng, state, rewards, dones = carry
            rng, key = jax.random.split(rng)
            actions = jax.random.randint(key, (params.n_drones,), 0, NUM_ACTIONS, dtype=jnp.int32)
            state, rewards, dones = self.step(rng, state, actions, params)
            return rng, state, rewards, dones

        carry = (key, state, jnp.zeros(params.n_drones), jnp.zeros(params.n_drones, dtype=jnp.bool))
        carry = jax.lax.fori_loop(0, num_steps, loop_body, carry)
        _, state, rewards, dones = carry
        return state, rewards, dones




    def format_action(self, *actions):
        return [['←', '↓', '→', '↑', 'X'][i] for i in actions]


if __name__ == "__main__":
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir('./jax_cache')

    # grid_size = 370
    # n_drones = 4096
    grid_size = 185
    n_drones = 1024
    drone_density = n_drones / (grid_size ** 2)

    params = DroneEnvParams(n_drones=n_drones, drone_density=drone_density)
    env = DeliveryDrones()
    # params = DroneEnvParams()
    # grid_size = int(jnp.ceil(jnp.sqrt(params.n_drones / params.drone_density)))
    rng = jax.random.PRNGKey(0)

    state = env.reset(rng, params, grid_size)
    # state = jax.jit(env.reset, static_argnums=(1, 2))(rng, params, grid_size)

    # num_steps = 1000
    # print(f'Start running {num_steps:,} steps...')
    # ts = timer()
    # run_steps_jit = jax.jit(env.run_steps, static_argnums=(2, 3))
    # state, rewards, dones = run_steps_jit(rng, state, params, 100)
    # rewards.block_until_ready()
    # te = timer()
    # print(f'... took {te-ts:.2f}s ({(te-ts)/num_steps:.5f}s/step)')
    # __import__('pdb').set_trace()

    step_jit = jax.jit(env.step, static_argnums=(3,))
    # step_jit = env.step
    num_steps = 100
    ts = timer()
    print(f'Start running {num_steps:,} steps sequentially...')
    for i in trange(num_steps):
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (params.n_drones,), 0, NUM_ACTIONS, dtype=jnp.int32)
        state, rewards, dones = step_jit(rng, state, actions, params)
    te = timer()
    print(f'... took {te-ts:.2f}s ({(te-ts)/num_steps:.4f}s/step)')
