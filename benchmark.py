import time
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from tabulate import tabulate

from env.v1.env import DeliveryDrones as DeliveryDronesV1
from env.v1.wrappers import WindowedGridView as WindowedGridViewV1
from env.v2.env import DeliveryDrones as DeliveryDronesV2
from env.v2.wrappers import WindowedGridView as WindowedGridViewV2
from env.v3.env import DeliveryDrones as DeliveryDronesV3
from env.v3.wrappers import WindowedGridView as WindowedGridViewV3


@dataclass
class Impl:
    name: str
    env: object
    desc: str


@dataclass
class Config:
    name: str
    params: dict


# CONFIG #
n_steps = 50
# drone_counts = [2048]
drone_counts = [32, 128, 512, 2048, 2048*4]

configs = [
    Config(
        name="DronesOnly",
        params={'packets_factor': 0, 'dropzones_factor': 0, 'stations_factor': 0, 'skyscrapers_factor': 0}
    ),
    Config(
        name="Default",
        params={}
    ),
    Config(
        name="HighDensity",
        params={'packets_factor': 4, 'dropzones_factor': 4, 'stations_factor': 4, 'skyscrapers_factor': 4}
    ),
]

impls = [
    Impl(
        name="v1",
        env=WindowedGridViewV1(DeliveryDronesV1(), radius=3),
        desc="original 2020 version"
    ),
    Impl(
        name="v2",
        env=WindowedGridViewV2(DeliveryDronesV2(), radius=3),
        desc="2020 grid-based version using constants instead of objects for what's on the map"
    ),
    Impl(
        name="v3",
        env=WindowedGridViewV3(DeliveryDronesV3(), radius=3),
        # env=DeliveryDronesV3(),
        desc="dict-based version"
    ),
]
# CONFIG #

results = []
reference_speeds = {}


def benchmark_implementation(imp, config, n_drones, n_steps):
    # Update config with the number of drones for this run
    config_params = {**config.params, 'n_drones': n_drones}
    imp.env.env_params.update(config_params)
    imp.env.reset()
    # print(imp.env.render(mode='ansi'))
    start_time = time.perf_counter()

    for _ in tqdm(range(n_steps), desc=f"{imp.name} {config.name} {n_drones} drones", leave=True):
        actions = {drone.index: imp.env.action_space.sample() for drone in imp.env.drones}
        imp.env.step(actions)
        # print(imp.env.render(mode='ansi'))

    total_time = time.perf_counter() - start_time
    mean_time = (total_time / n_steps) * 1000  # Convert to ms per step and scale for per 1000 steps
    sps = n_steps / total_time

    # Identify reference time for comparison
    key = (config.name, n_drones)
    if imp.name == "v1":
        reference_speeds[key] = mean_time
    percent_diff = (reference_speeds[key] / mean_time) if key in reference_speeds else 0

    results.append([
        imp.name, config.name, n_drones, f"{imp.env.side_size}x{imp.env.side_size}", f"{mean_time:.1f} spKs",
        f"{sps:.1f} sps", f"{percent_diff:.2f}"
    ])


if __name__ == '__main__':
    for imp in impls:
        for config in configs:
            for n_drones in drone_counts:
                benchmark_implementation(imp, config, n_drones, n_steps)

    # Print results as a table
    print(
        tabulate(results, headers=['Implementation', 'Config', 'Drones', "Size", 'Speed', 'Speed', 'Speedup from V1'])
    )
    print("=" * 15)
    for imp in impls:
        print(f"{imp.name}: {imp.desc}")
    print("=" * 15)
    print(f"{n_steps:,} steps per run.")
