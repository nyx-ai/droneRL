# 🚁 DroneRL

DroneRL is a reinforcement learning 2D grid world environment in which agents can be trained for optimal package delivery.

This environment is implemented in both PyTorch and JAX.

![output](https://github.com/user-attachments/assets/babedf9d-d062-48f9-9e5e-37d939581a4c)

*Player 0 is an agent trained for 5M steps, other players act randomly*

🥇 There is a public leaderboard on AICrowd for this problem, check it out on [AIcrowd](https://www.aicrowd.com/challenges/dronerl/leaderboards).

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

### JAX
Everything is identical to PyTorch above, except use the the JAX training script
```bash
python train_jax.py
```

## 🏭 Performance
### Torch vs. JAX

![torch_vs_jax](https://github.com/user-attachments/assets/1158cebe-9c62-4a3e-ae85-68da03c4081b)

Due to warmup and compilation, JAX will be slower when running a small number of steps, but then should take over. Note that the torch implementation has been very slightly adjusted in order to make this comparison fair. The resulting eval reward is within error margins between the two implementations.

### Scaling up number of envs
🚧

### JAX on accelerators
The benefit of the JAX implementation is that the code runs on both GPUs and TPUs end-to-end, meaning both environment and agent are leveraging accelerators. In order for this to work the JAX environment step function is fully vectorized and the whole training loop makes use of loop unrolling via [JAX scans](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html), thereby minimizing host-accelerator communication.

🚧

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
