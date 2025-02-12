import torch
import numpy as np
import pytest
import jax
import jax.random
import tempfile

from torch_impl.agents.dqn import DQNAgent as PyDQNAgent
from torch_impl.env.wrappers import WindowedGridView
from torch_impl.env.env import DeliveryDrones as PyDeliveryDrones
from torch_impl.agents.dqn import BaseDQNFactory
from jax_impl.env.env import DroneEnvParams, DeliveryDrones
from jax_impl.agents.dqn import DQNAgent, DQNAgentParams


@pytest.fixture
def jax_dqn_agent():
    return DQNAgent()


@pytest.fixture
def jax_obs():
    params = DroneEnvParams()
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)
    env_state = env.reset(rng, params)
    obs = env.get_obs(env_state, params)[0]
    return obs.ravel()


def test_dqn_agent_save(jax_dqn_agent, jax_obs):
    env_params = DroneEnvParams()
    ag_params = DQNAgentParams()
    rng = jax.random.PRNGKey(42)
    ag_state = jax_dqn_agent.reset(rng, ag_params, env_params)
    py_env_params = {
        'n_drones': env_params.n_drones,
        'drone_density': 0.3,
        'packets_factor': 1,
        'dropzones_factor': 1,
        'stations_factor': 1,
        'skyscrapers_factor': 1
    }
    py_env = WindowedGridView(PyDeliveryDrones(py_env_params), radius=env_params.window_radius)
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tf:
        jax_dqn_agent.save_as_torch(tf.file.name, ag_state, ag_params, env_params)
        py_agent = PyDQNAgent(
            env=py_env,
            dqn_factory=BaseDQNFactory.from_checkpoint(tf.file.name),
            gamma=0.99,
            epsilon_start=0,  # No exploration
            epsilon_decay=1,
            epsilon_end=0,
            memory_size=1000,
            batch_size=32,
            target_update_interval=100
        )
        with torch.no_grad():
            py_out = py_agent.qnetwork([torch.from_numpy(np.asarray(jax_obs).copy())])[0]
        jax_out = ag_state.qnetwork.apply(ag_state.qnetwork_params, jax_obs)
        assert torch.allclose(py_out, torch.from_numpy(np.asarray(jax_out).copy()))
