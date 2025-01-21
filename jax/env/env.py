from typing import Optional, Tuple, Callable, Literal
import jax
import jax.numpy as jnp
import jax.random
import jax.nn
from tqdm import trange
from flax.struct import dataclass
from timeit import default_timer as timer

from .constants import Action, Object


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
    wrapper: Literal['window', 'compass'] = 'window'
    window_radius: int = 3


@dataclass
class DroneEnvState:
    ground: jnp.ndarray
    air_x: jnp.ndarray
    air_y: jnp.ndarray
    carrying_package: jnp.ndarray
    charge: jnp.ndarray


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
            p_choice &= ~exclude_mask
        p_choice = p_choice.ravel()

        # old method
        # pos = jax.random.choice(key, params.grid_size ** 2, shape=fill_values.shape, p=p_choice, replace=False)

        # new method
        # noise = jax.random.gumbel(key, shape=(params.grid_size ** 2,))
        noise = jax.random.uniform(key, shape=(params.grid_size ** 2,))
        scores = jnp.log(p_choice) + noise
        # _, pos = jax.lax.top_k(scores, k=fill_values.shape[0])
        _, pos = jax.lax.approx_max_k(scores, fill_values.shape[0])

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
        p_choice = jnp.ones((params.grid_size, params.grid_size), dtype=jnp.bool)
        p_choice = p_choice.at[x_pos, y_pos].set(False)
        if exclude is not None:
            p_choice &= ~exclude
        p_choice = p_choice.ravel()
        # pos = jax.random.choice(key, params.grid_size ** 2, shape=(params.n_drones,), p=p_choice, replace=False)

        # new method
        # noise = jax.random.gumbel(key, shape=(params.grid_size ** 2,))
        noise = jax.random.uniform(key, shape=(params.grid_size ** 2,))
        scores = jnp.log(p_choice) + noise
        # _, pos = jax.lax.top_k(scores, k=params.n_drones)
        _, pos = jax.lax.approx_max_k(scores, params.n_drones)

        random_x_pos = pos // params.grid_size
        random_y_pos = pos % params.grid_size
        x_pos = jnp.where(x_pos == -1, random_x_pos, x_pos)
        y_pos = jnp.where(y_pos == -1, random_y_pos, y_pos)
        return x_pos, y_pos


    def reset(self, key: jax.Array, params: DroneEnvParams):
        # validations
        num_packets = params.packets_factor * params.n_drones
        num_dropzones = params.dropzones_factor * params.n_drones
        num_stations = params.stations_factor * params.n_drones
        num_skyscrapers = params.skyscrapers_factor * params.n_drones
        num_objects = num_packets + num_skyscrapers + num_dropzones + num_stations
        if num_objects > params.grid_size ** 2:
            raise ValueError(
                    f'Grid supports only {params.grid_size**2:,} positions but {num_objects:,} objects ' \
                    f'({num_dropzones:,} dropzones, {num_stations:,} charging stations, {num_packets:,} packages, ' \
                    f'{num_skyscrapers:,} skyscrapers) were attempted to be placed.')
        elif params.n_drones > params.grid_size ** 2:
            raise ValueError(
                    f'Grid supports only {params.grid_size**2:,} positions but {params.n_drones:,} drones were ' \
                    'attempted to be placed.')
        # spawn objects
        ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_packets, dtype=jnp.int8) * Object.PACKET.value, params)
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_dropzones, dtype=jnp.int8) * Object.DROPZONE.value, params)
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_stations, dtype=jnp.int8) * Object.STATION.value, params)
        key, spawn_key = jax.random.split(key)
        ground = self.spawn(spawn_key, ground, jnp.ones(num_skyscrapers, dtype=jnp.int8) * Object.SKYSCRAPER.value, params)
        # spawn drones
        air_x = -1 * jnp.ones(params.n_drones, dtype=jnp.int32)  # set positions to -1 for now
        air_y = -1 * jnp.ones(params.n_drones, dtype=jnp.int32)  # set positions to -1 for now
        key, spawn_key = jax.random.split(key)
        air_x, air_y = self.spawn_air(spawn_key, air_x, air_y, params, exclude=(ground == Object.SKYSCRAPER))

        # check if drones picked up package
        carrying_package = (ground[air_y, air_x] == Object.PACKET)

        # remove packages that were picked up by respawned drones
        mask = jnp.zeros_like(ground, dtype=jnp.bool)
        mask = mask.at[air_y, air_x].set(carrying_package)
        ground *= ~mask
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
        dy = jnp.where(actions == Action.UP, -1, jnp.where(actions == Action.DOWN, 1, 0))
        dx = jnp.where(actions == Action.LEFT, -1, jnp.where(actions == Action.RIGHT, 1, 0))
        new_y = state.air_y + dy
        new_x = state.air_x + dx

        # check off board
        off_board = (new_y < 0) | (new_y >= params.grid_size) | (new_x < 0) | (new_x >= params.grid_size)

        # check skyscrapers
        on_board_y = jnp.clip(new_y, 0, params.grid_size - 1)
        on_board_x = jnp.clip(new_x, 0, params.grid_size - 1)
        collided_skyscrapers = (state.ground[on_board_y, on_board_x] == Object.SKYSCRAPER) & ~off_board

        # collisions between drones
        new_pos = jnp.stack([new_x, new_y], axis=1)
        _, inverse_indices, counts = jnp.unique(
                new_pos,
                axis=0,
                return_inverse=True,
                return_counts=True,
                size=params.n_drones)
        collisions = counts[inverse_indices] > 1
        collided = off_board | collided_skyscrapers | collisions

        # handle charge
        is_charging = (state.ground[new_y, new_x] == Object.STATION) & (~collided) & (state.charge < 100)
        is_discharging = ~is_charging & (~collided)
        charge = (state.charge + is_charging * params.charge).clip(0, 100)
        charge = (charge - is_discharging * params.discharge).clip(0, 100)
        out_of_charge = charge == 0

        # compute who survived
        dones = collided | out_of_charge
        survivors = ~dones

        # compute new charge
        charge = jnp.where(dones, 100, charge)  # start with a full charge again after done

        # check whether we can pick up a package
        surivors_picked_up = (state.ground[new_y, new_x] == Object.PACKET) & survivors & ~state.carrying_package  # surivors may pick up packages at new locations
        mask = jnp.zeros_like(state.ground, dtype=jnp.bool)
        mask = mask.at[new_y, new_x].set(surivors_picked_up)
        ground = state.ground * ~mask
        carrying_package = state.carrying_package & survivors  # dead drones lose their packages
        carrying_package |= surivors_picked_up   # add packages that were newly picked up by survivors

        # check whether we can deliver a package
        is_at_dropoff_position = (state.ground[new_y, new_x] == Object.DROPZONE) & survivors
        delivered = is_at_dropoff_position & state.carrying_package
        carrying_package &= ~delivered  # remove delivered packages

        # respawn delivered packages and packages that were carried by dead drones
        key, spawn_key = jax.random.split(key)
        num_packets = params.packets_factor * params.n_drones
        fill_values = jnp.zeros(num_packets, dtype=jnp.int8)
        # trick: if no deliveries took place we spawn a "0", else Object.PACKET
        done_while_carrying_package = dones & state.carrying_package
        fill_values = fill_values.at[:len(delivered)].set((delivered | done_while_carrying_package) * Object.PACKET.value)
        ground = self.spawn(spawn_key, ground, fill_values, params)

        # respawn dropzones
        num_dropzones = params.packets_factor * params.n_drones
        fill_values = jnp.zeros(num_dropzones, dtype=jnp.int8)
        fill_values = fill_values.at[:len(delivered)].set(delivered * Object.DROPZONE.value)
        mask = jnp.zeros_like(state.ground, dtype=jnp.bool)
        mask = mask.at[new_y, new_x].set(delivered)
        ground = ground * ~mask
        ground = self.spawn(spawn_key, ground, fill_values, params)

        # compute rewards
        rewards = jnp.zeros(params.n_drones, dtype=jnp.float32)
        rewards += params.crash_reward * dones
        rewards += params.pickup_reward * surivors_picked_up
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
                exclude=(state.ground == Object.SKYSCRAPER))

        # potentially pick up newly spawned packages
        package_mask = (ground == Object.PACKET)
        can_pick_up = package_mask[new_x, new_y]
        picked_up_after_respawn = can_pick_up & dones
        carrying_package |= picked_up_after_respawn

        # remove packages that were picked up by respawned drones
        mask = jnp.zeros_like(state.ground, dtype=jnp.bool)
        mask = mask.at[new_x, new_y].set(picked_up_after_respawn)
        ground *= ~mask

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
            agent_action: Callable,
            ) -> Tuple[DroneEnvState, jnp.ndarray, jnp.ndarray]:

        def loop_body(i, carry):
            rng, state, rewards, dones = carry
            rng, key = jax.random.split(rng)
            action_keys = jax.random.split(rng, params.n_drones)
            actions = jax.vmap(agent_action)(action_keys)
            state, rewards, dones = self.step(rng, state, actions, params)
            return rng, state, rewards, dones

        carry = (key, state, jnp.zeros(params.n_drones), jnp.zeros(params.n_drones, dtype=jnp.bool))
        carry = jax.lax.fori_loop(0, num_steps, loop_body, carry)
        _, state, rewards, dones = carry
        return state, rewards, dones

    def get_obs(self, env_state: DroneEnvState, params: DroneEnvParams):
        if params.wrapper != 'window':
            # currently only support for wrapper
            raise NotImplementedError
        radius = params.window_radius
        padded = jnp.pad(env_state.ground, radius, mode='constant', constant_values=Object.SKYSCRAPER)  # pad with skyscrapers
        x_pos, y_pos = env_state.air_x + radius, env_state.air_y + radius
        padded = padded.at[y_pos, x_pos].add(100)  # required to get drone positions
        x_indices = x_pos[:, None] + jnp.arange(-radius, radius + 1, dtype=jnp.int32)[None, :]  # (n, 1) + (1, 2r+1) => (n, 2r+1)
        y_indices = y_pos[:, None] + jnp.arange(-radius, radius + 1, dtype=jnp.int32)[None, :]
        obs_org = padded[y_indices[:, :, None], x_indices[:, None, :]]  # => (n, 2r+1, 2r+1)

        # re-map some of the classes for one-hot encoding
        # currently: skyscraper: 2, station: 3, dropzone: 4, packages: 5
        # new: drone pos: 0, packages: 1, dropzones: 2, station: 3, charge: 4, skyscrapers/wall: 5
        obj_only = obs_org % 100
        obj_only = jnp.where(obj_only == Object.PACKET, 1,   # packages -> 1
                  jnp.where(obj_only == Object.SKYSCRAPER, 5,   # skyscrapers -> 5
                  jnp.where(obj_only == Object.DROPZONE, 2,   # dropzones -> 2
                  jnp.where(obj_only == 0, 10,   # empty -> arbitrary value > 6
                           obj_only))))  # default case

        # generate one-hot encoding
        obs = jax.nn.one_hot(obj_only, 6, dtype=jnp.bool)
        obs = obs.at[:, :, :, 0].set(obs_org >= 100)

        # mark package if carrying
        obs = obs.at[:, radius, radius, 1].set(obs[:, radius, radius, 1] | env_state.carrying_package)

        # mark charge status
        obs = obs.astype(jnp.float32)
        obs = obs.at[:, radius, radius, 4].set(env_state.charge / 100.0)
        return obs

    def format_action(self, *actions):
        return [['←', '↓', '→', '↑', 'X'][i] for i in actions]


if __name__ == "__main__":
    from jax.experimental.compilation_cache import compilation_cache as cc
    import statistics
    cc.set_cache_dir('./jax_cache')
    import sys; sys.path.append('..')
    from agents.rand import RandomAgent
    from agents.dqn import DQNAgent

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
    agent = RandomAgent()
    # agent = DQNAgent()

    # #######################
    # # jit + Python for-loop
    # #######################
    # state = env.reset(rng, params)
    state = jax.jit(env.reset, static_argnums=(1,))(rng, params)
    step_jit = jax.jit(env.step, static_argnums=(3,))
    get_obs_jit = jax.jit(env.get_obs)
    # step_jit = env.step
    repeats = 5
    skip = 1 if repeats > 1 else 0  # first run is usually a bit slower (warmup)
    num_steps = 100
    print(f'Running {num_steps:,} steps {repeats} times (skipping {skip} first runs)...')
    times = []
    for _ in range(repeats):
        ts = timer()
        for i in trange(num_steps):
            rng, key = jax.random.split(rng)
            action_keys = jax.random.split(key, params.n_drones)
            # actions = jax.vmap(agent.act)(action_keys)
            actions = jax.vmap(agent.act)(action_keys)
            state, rewards, dones = step_jit(rng, state, actions, params)
            obs = get_obs_jit(state)

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

    # ############
    # # jit + fori
    # ############
    # # state = env.reset(rng, params, grid_size)
    # state = jax.jit(env.reset, static_argnums=(1,))(rng, params)
    # num_steps = 100
    # repeats = 5
    # skip = 1 if repeats > 1 else 0  # first run is usually a bit slower (warmup)
    # print(f'Start running {num_steps:,} steps {repeats} times...')
    # times = []
    # run_steps_jit = jax.jit(env.run_steps, static_argnums=(2, 3, 4))
    # for _ in trange(repeats):
    #     ts = timer()
    #     state, rewards, dones = run_steps_jit(rng, state, params, num_steps, agent.act)
    #     rewards.block_until_ready()
    #     times.append(timer() - ts)
    # mean, std = statistics.mean(times[skip:]), statistics.stdev(times[skip:])
    # print(f'... jit+fori took {mean:.2f}s (±{std:.3f}) or {1000*mean/(num_steps):.4f}s/1k steps (±{1000*std/num_steps:.4f})')
    # __import__('pdb').set_trace()

    # ##################
    # # jit + vmap + fori
    # ##################
    # num_envs = 4
    # keys = jax.random.split(rng, num_envs)
    # state = jax.jit(jax.vmap(env.reset, in_axes=[0, None]), static_argnums=(1,))(keys, params)
    # num_steps = 100
    # repeats = 5
    # skip = 1 if repeats > 1 else 0  # first run is usually a bit slower (warmup)
    # print(f'Start running {num_steps:,} steps with {num_envs:,} envs for {repeats} times...')
    # run_steps_jit = jax.jit(jax.vmap(env.run_steps, in_axes=[0, 0, None, None]), static_argnums=(2, 3))
    # times = []
    # for _ in trange(repeats):
    #     ts = timer()
    #     rng, _ = jax.random.split(rng)
    #     keys = jax.random.split(rng, num_envs)
    #     state, rewards, dones = run_steps_jit(keys, state, params, num_steps)
    #     rewards.block_until_ready()
    #     times.append(timer() - ts)
    # mean, std = statistics.mean(times[skip:]), statistics.stdev(times[skip:])
    # print(f'... jit+vmap+fori took {mean:.2f}s (±{std:.3f}) or {1000*mean/(num_steps*num_envs):.4f}s/1k steps (±{1000*std/(num_steps * num_envs):.4f})')
    # __import__('pdb').set_trace()

    ############
    # pmap+vmap+fori
    ############
    # from flax.training.common_utils import shard
    # num_envs = 1024
    # keys = jax.random.split(rng, num_envs)
    # state = jax.jit(jax.vmap(env.reset, in_axes=[0, None]), static_argnums=(1,))(keys, params)
    # num_devices = jax.device_count()
    # state = shard(state)
    # num_steps = 100
    # repeats = 5
    # skip = 1 if repeats > 1 else 0  # first run is usually a bit slower (warmup)
    # print(f'Start running {num_steps:,} steps with {num_envs:,} envs for {repeats} times...')
    # times = []
    # p_run_steps = jax.pmap(
    #         jax.vmap(env.run_steps, in_axes=[0, 0, None, None]),
    #         static_broadcasted_argnums=(2, 3))
    # for _ in trange(repeats):
    #     ts = timer()
    #     rng, _ = jax.random.split(rng)
    #     keys = jax.random.split(rng, num_envs)
    #     keys = shard(keys)
    #     state, rewards, dones = p_run_steps(keys, state, params, num_steps)
    #     rewards.block_until_ready()
    #     times.append(timer() - ts)
    # mean, std = statistics.mean(times[skip:]), statistics.stdev(times[skip:])
    # print(f'... pmap+vmap+fori took {mean:.2f}s (±{std:.3f}) or {1000*mean/(num_steps*num_envs):.4f}s/1k steps (±{1000*std/(num_steps * num_envs):.4f})')
    # __import__('pdb').set_trace()
