# 🚁 DroneRL

DroneRL is a reinforcement learning 2D grid world environment in which agents can be trained for optimal package delivery.

This environment is implemented in both PyTorch and JAX.

![output](https://github.com/user-attachments/assets/babedf9d-d062-48f9-9e5e-37d939581a4c)

In this video, player 0 is our agent that has been trained for 5M steps, other players act randomly. The reward sum shows the cumulative reward granted to the players as they are delivering packages. In this video a drone receives +1.0 reward for delivering a package, -0.1 for charging, and -1.0 for crashing or running out of battery.

🥇 There is a public leaderboard on AICrowd for this problem, check it out on [AIcrowd](https://www.aicrowd.com/challenges/dronerl/leaderboards).

## Changelog
The project has evolved quite a bit throughout the years, and has experienced some breaking changes.
* v0.2 (03/2025)
  * Added JAX implementation
  * End-to-end training scripts for both PyTorch and JAX
  * Improved video visualization
  * Updated dependencies
  * Added tests
* v0.1.1 (03/2024)
  * Updated dependencies/Colab compatibility
* v0.1 (02/2020)
  * First version used for workshops at AMLD 2020 and IIIT-H

## 📦 Install
This code was tested with Python `3.11`.

In order to use the PyTorch implementation, run
```bash
pip install -r torch_impl/requirements.txt
```

For JAX use
```bash
pip install -r jax_impl/requirements.cpu.txt
```
Depending on the platform (e.g. GPU or TPU) you'll also have to install `tpulib` along side (see [here](https://docs.jax.dev/en/latest/installation.html))

## 🚀 Getting started
### PyTorch
To train your first agent run
```bash
python train_torch.py
```
This will train a DQN agent for 1k steps and show you the eval reward (i.e. average reward over 5 seeds). Training generally works if the eval reward is higher than the reward of a random agent.

For a list of available parameters run
```bash
python train_torch.py --help
```

Currently PyTorch does not support training in multiple environments (i.e. `--num_envs` > 1).

### JAX
JAX uses similar arguments and the same default as the above PyTorch training script. You can train an agent in JAX using
```bash
python train_jax.py
```


## 🏭 Performance
### Torch vs. JAX

![torch_vs_jax](https://github.com/user-attachments/assets/1158cebe-9c62-4a3e-ae85-68da03c4081b)

Due to warmup and compilation, JAX will be slower when running a small number of steps, but then should take over. Note that the torch implementation has been very slightly adjusted in order to make this comparison fair. The resulting eval reward is within error margins between the two implementations.

### JAX on accelerators
The benefit of the JAX implementation is that the code runs on both GPUs and TPUs end-to-end, meaning both environment and agent are leveraging accelerators. In order for this to work the JAX environment step function is fully vectorized and the whole training loop makes use of loop unrolling via [JAX scans](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html), thereby minimizing host-accelerator communication.

| Hardware           | (16, 16) dense network (obs/s) | (128, 64) dense network (obs/s) |
|--------------------|------------------------------:|---------------------------------:|
| CPU (colab)        |                         8,494 |                            2,805 |
| GPU T4             |                         4,113 |                            3,699 |
| TPU v2-8           |                         3,186 |                            3,121 |
| TPU v3-8           |                         3,843 |                            3,641 |
| Mac M2 Pro (CPU)   |                        14,739 |                            5,017 |

As you can see, leveraging accelerators only really makes sense when training larger networks. You might even be able to train faster on CPU when training a very small network architecture.

### Scaling up envs and env sharding for JAX
If you have multiple devices available (e.g. a TPU v3-8 has 8 devices), you may use training with sharded envs. This increases the number of observations you can generate as you're making use of all available devices.

![num_envs](https://github.com/user-attachments/assets/5c9215ac-3207-464e-bea9-9e15f1b12e55)

Note that as you generate more observations in each training step you may also want to increase the batch size and learning rate in order for efficient learning to happen. In order to use sharding across envs, use the `--num_envs` and `--use_sharding` arguments in the `train_jax.py` script. Note that the number of envs needs to be divisible by the number of devices.

## Credits
Part of this work was supported by the [EPFL Extension School](http://exts.epfl.ch/) and [AIcrowd](http://aicrowd.com/).

This more optimized version of DroneRL was implemented by [@MasterScrat](https://github.com/masterScrat) and [@mar-muel](https://github.com/mar-muel/), but over the years many people have contributed:
* [@spMohanty](https://github.com/spmohanty)
* [@pacm](https://github.com/pacm)
* [@metataro](https://github.com/metataro)

## License
* 16ShipCollection by master484 under Public Domain / CC0
* Inconsolata-Bold by Raph Levien under Open Font License
* Press Start 2P by Cody Boisclair under SIL Open Font License 1.1
