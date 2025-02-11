import pytest
import jax
import jax.numpy as jnp
import jax.random

from env.env import DroneEnvState, DroneEnvParams, DeliveryDrones
from env.constants import Object, Action
from agents.rand import RandomAgent
from agents.dqn import DQNAgent, DQNAgentParams, DQNAgentState


@pytest.fixture
def rand_agent():
    return RandomAgent()


@pytest.fixture
def dqn_agent():
    return DQNAgent()


@pytest.fixture
def obs():
    params = DroneEnvParams()
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)
    env_state = env.reset(rng, params)
    obs = env.get_obs(env_state, params)[0]
    return obs.ravel()


def test_dqn_agent_act(dqn_agent, obs):
    ag_params = DQNAgentParams()
    env_params = DroneEnvParams()
    rng = jax.random.PRNGKey(0)
    ag_state = dqn_agent.reset(rng, ag_params, env_params)
    action = dqn_agent.act(rng, obs, ag_state)
    assert action.shape == ()
    assert action.dtype == jnp.int32
