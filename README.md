# 🚁 DroneRL

DroneRL is a reinforcement learning 2D grid world environment in which agents can be trained for optimal package delivery.

This environment is implemented in both PyTorch and JAX.

![output](https://github.com/user-attachments/assets/f87800a4-3414-4949-bd05-e567a51da9b2)
_Player 0 is an agent trained for 5M steps, other players act randomly_

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
