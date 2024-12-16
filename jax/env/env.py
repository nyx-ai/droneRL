from typing import Optional, Dict, Tuple, Union
import jax
from jax.lax import pbroadcast
import jax.numpy as jnp
import jax.random
from tqdm import trange
from flax.struct import dataclass
from timeit import default_timer as timer


@dataclass
class DroneEnvParams:
    grid_size: int = 8
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
class DroneEnvState:
    ground: jnp.ndarray
    air_x: jnp.ndarray
    air_y: jnp.ndarray
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


class DeliveryDrones:
    def spawn(
            self,
            key: jnp.ndarray,
            grid: jnp.ndarray,
            fill_values: jnp.ndarray,
            params: DroneEnvParams,
            exclude_mask: Optional[jnp.ndarray] = None,
            ) -> jnp.ndarray:
        p_choice = (grid == 0)
        if exclude_mask:
            p_choice = (grid == 0 & ~exclude_mask)
        else:
            p_choice = (grid == 0)
        p_choice = p_choice.astype(jnp.float32).ravel()
        pos = jax.random.choice(key, params.grid_size ** 2, shape=fill_values.shape, p=p_choice)
        random_x_pos = pos // params.grid_size
        random_y_pos = pos % params.grid_size
        grid = grid.at[random_x_pos, random_y_pos].set(fill_values)
        return grid

    def spawn_air(
            self,
            key: jnp.ndarray,
            x_pos: jnp.ndarray,
            y_pos: jnp.ndarray,
            params: DroneEnvParams,
            exclude: Optional[jnp.ndarray] = None,
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        p_choice = jnp.ones((params.grid_size, params.grid_size), dtype=jnp.float32)
        p_choice = p_choice.at[x_pos, y_pos].set(0)
        if exclude is not None:
            p_choice = jnp.where(exclude, 0, p_choice)
        p_choice = p_choice.ravel()
        pos = jax.random.choice(key, params.grid_size ** 2, shape=(params.n_drones,), p=p_choice)
        random_x_pos = pos // params.grid_size
        random_y_pos = pos % params.grid_size
        x_pos = jnp.where(x_pos == -1, random_x_pos, x_pos)
        y_pos = jnp.where(y_pos == -1, random_y_pos, y_pos)
        return x_pos, y_pos


    def reset(self, key: jax.Array, params: DroneEnvParams):
        # spawn objects
        ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
        num_packets = params.packets_factor * params.n_drones
        num_dropzones = params.dropzones_factor * params.n_drones
        num_stations = params.stations_factor * params.n_drones
        num_skyscrapers = params.skyscrapers_factor * params.n_drones
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_packets, dtype=jnp.int8) * OBJ_PACKET, params)
        key, spawn_key = jax.random.split(key)
        ground= self.spawn(spawn_key, ground, jnp.ones(num_dropzones, dtype=jnp.int8) * OBJ_DROPZONE, params)
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_stations, dtype=jnp.int8) * OBJ_STATION, params)
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_skyscrapers, dtype=jnp.int8) * OBJ_SKYSCRAPER, params)
        # spawn drones
        air_x = -1 * jnp.ones(params.n_drones, dtype=jnp.int32)  # set positions to -1 for now
        air_y = -1 * jnp.ones(params.n_drones, dtype=jnp.int32)  # set positions to -1 for now
        key, spawn_key = jax.random.split(key)
        air_x, air_y = self.spawn_air(spawn_key, air_x, air_y, params, exclude=(ground == OBJ_SKYSCRAPER))

        # check if drones picked up package
        carrying_package = (ground[air_y, air_x] == OBJ_PACKET)

        # remove packages that were picked up by respawned drones
        mask = jnp.zeros_like(ground, dtype=jnp.bool)
        mask = mask.at[air_y, air_x].set(carrying_package)
        ground = jnp.where(mask, 0, ground)
        state = DroneEnvState(
                air_x=air_x,
                air_y=air_y,
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
             ) -> Tuple[DroneEnvState, jnp.ndarray, jnp.ndarray]:
        # compute new position of drones
        dy = jnp.where(actions == ACTION_UP, -1, jnp.where(actions == ACTION_DOWN, 1, 0))
        dx = jnp.where(actions == ACTION_LEFT, -1, jnp.where(actions == ACTION_RIGHT, 1, 0))
        new_y = state.air_y + dy
        new_x = state.air_x + dx

        # check off board
        off_board = (new_y < 0) | (new_y >= params.grid_size) | (new_x < 0) | (new_x >= params.grid_size)

        # check skyscrapers
        on_board_y = jnp.clip(new_y, 0, params.grid_size - 1)
        on_board_x = jnp.clip(new_x, 0, params.grid_size - 1)
        collided_skyscrapers = (state.ground[on_board_y, on_board_x] == OBJ_SKYSCRAPER) & ~off_board

        # collisions between drones
        new_pos = jnp.stack([new_x, new_y], axis=1)
        uq_pos, inverse_indices, counts = jnp.unique(
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
        fill_values = jnp.zeros(num_packets, dtype=jnp.int8)
        # trick: if no deliveries took place we spawn a "0", else OBJ_PACKET
        fill_values = fill_values.at[:len(delivered)].set(delivered * OBJ_PACKET)
        ground = self.spawn(spawn_key, ground, fill_values, params)

        # handle charge
        is_charging = (state.ground[new_y, new_x] == OBJ_STATION) & survivors & (state.charge < 100)
        is_discharging = ~is_charging & survivors
        charge = (state.charge + is_charging * params.charge).clip(0, 100)
        charge = (charge - is_discharging * params.discharge).clip(0, 100)
        out_of_charge = charge == 0
        charge = jnp.where(out_of_charge, 100, charge)  # we start with full charge again after out of charge
        dones = dones | out_of_charge
        survivors = ~dones

        # compute rewards
        rewards = jnp.zeros(params.n_drones, dtype=jnp.float32)
        rewards += params.crash_reward * dones
        rewards += params.pickup_reward * picked_up
        rewards += params.delivery_reward * delivered
        rewards += params.charge_reward * is_charging

        # respawn dead players
        new_x = jnp.where(dones, -1, new_x)
        new_y = jnp.where(dones, -1, new_y)
        key, spawn_key = jax.random.split(key)
        new_x, new_y = self.spawn_air(
                spawn_key,
                new_x,
                new_y,
                params,
                exclude=(state.ground == OBJ_SKYSCRAPER))

        # potentially pick up newly spawned packages
        package_mask = (ground == OBJ_PACKET)
        can_pick_up = package_mask[new_x, new_y]
        picked_up = can_pick_up & dones  # all the newly spawned players that can now pick up a package
        carrying_package |= picked_up

        # remove packages that were picked up by respawned drones
        mask = jnp.zeros_like(state.ground, dtype=jnp.bool)
        mask = mask.at[new_x, new_y].set(picked_up)
        ground = jnp.where(mask, 0, ground)

        # update state
        state = state.replace(
                air_x=new_x,
                air_y=new_y,
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
            ) -> Tuple[DroneEnvState, jnp.ndarray, jnp.ndarray]:

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
    import statistics
    cc.set_cache_dir('./jax_cache')

    # grid_size = 8
    # n_drones = 3
    # grid_size = 370
    # n_drones = 4096
    grid_size = 185
    n_drones = 1024
    # grid_size = 64
    # n_drones = int(0.03 * (grid_size ** 2))

    drone_density = n_drones / (grid_size ** 2)
    print(f'Num drones: {n_drones}, grid: {grid_size}x{grid_size}, drone density: {drone_density:.2f}')

    params = DroneEnvParams(n_drones=n_drones, grid_size=grid_size)
    env = DeliveryDrones()
    # params = DroneEnvParams()
    # grid_size = int(jnp.ceil(jnp.sqrt(params.n_drones / params.drone_density)))
    rng = jax.random.PRNGKey(0)

    ############
    # jit + fori
    ############
    # state = env.reset(rng, params, grid_size)
    state = jax.jit(env.reset, static_argnums=(1,))(rng, params)
    num_steps = 1000
    repeats = 5
    skip = 1 if repeats > 1 else 0  # first run is usually a bit slower (warmup)
    print(f'Start running {num_steps:,} steps {repeats} times...')
    times = []
    for _ in trange(repeats):
        ts = timer()
        run_steps_jit = jax.jit(env.run_steps, static_argnums=(2, 3))
        state, rewards, dones = run_steps_jit(rng, state, params, 100)
        rewards.block_until_ready()
        times.append(timer() - ts)
    mean, std = statistics.mean(times[skip:]), statistics.stdev(times[skip:])
    print(f'... jit+fori took {mean:.2f}s (±{std:.3f}) or {1000*mean/(num_steps):.4f}s/1k steps (±{1000*std/num_steps:.4f})')

    ############
    # jit + vmap
    ############
    num_envs = 4
    keys = jax.random.split(rng, num_envs)
    state = jax.jit(jax.vmap(env.reset, in_axes=[0, None]), static_argnums=(1,))(keys, params)
    num_steps = 1000
    repeats = 5
    skip = 1 if repeats > 1 else 0  # first run is usually a bit slower (warmup)
    print(f'Start running {num_steps:,} steps with {num_envs:,} envs for {repeats} times...')
    times = []
    for _ in trange(repeats):
        ts = timer()
        rng, _ = jax.random.split(rng)
        keys = jax.random.split(rng, num_envs)
        run_steps_jit = jax.jit(jax.vmap(env.run_steps, in_axes=[0, 0, None, None]), static_argnums=(2, 3))
        state, rewards, dones = run_steps_jit(keys, state, params, 100)
        rewards.block_until_ready()
        times.append(timer() - ts)
    mean, std = statistics.mean(times[skip:]), statistics.stdev(times[skip:])
    print(f'... jit+fori+vmap took {mean:.2f}s (±{std:.3f}) or {1000*mean/(num_steps*num_envs):.4f}s/1k steps (±{1000*std/(num_steps * num_envs):.4f})')

    #######################
    # jit + Python for-loop
    #######################
    # state = env.reset(rng, params)
    state = jax.jit(env.reset, static_argnums=(1,))(rng, params)
    step_jit = jax.jit(env.step, static_argnums=(3,))
    # step_jit = env.step
    repeats = 5
    skip = 1 if repeats > 1 else 0  # first run is usually a bit slower (warmup)
    num_steps = 1000
    print(f'Running {num_steps:,} steps {repeats} times (skipping {skip} first runs)...')
    times = []
    for _ in range(repeats):
        ts = timer()
        for i in trange(num_steps):
            rng, key = jax.random.split(rng)
            actions = jax.random.randint(key, (params.n_drones,), 0, NUM_ACTIONS, dtype=jnp.int32)
            state, rewards, dones = step_jit(rng, state, actions, params)

            # # tracing
            # state, rewards, dones = step_jit(rng, state, actions, params)
            # state, rewards, dones = step_jit(rng, state, actions, params)
            # with jax.profiler.trace("jax-trace-v2"):
            #     state, rewards, dones = step_jit(rng, state, actions, params)
            #     rewards.block_until_ready()
            # __import__('pdb').set_trace()
        te = timer()
        times.append(te - ts)
    mean, std = statistics.mean(times[skip:]), statistics.stdev(times[skip:])
    print(f'... jit+for-loop took {mean:.2f}s (±{std:.3f}) or {1000*mean/num_steps:.4f}s/1k steps (±{1000*std/num_steps:.4f})')
    __import__('pdb').set_trace()
