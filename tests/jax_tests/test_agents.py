import pytest
import jax
import jax.numpy as jnp
import jax.random
import tempfile

from jax_impl.env.env import DroneEnvParams, DeliveryDrones
from jax_impl.agents.rand import RandomAgent
from jax_impl.agents.dqn import DQNAgent, DQNAgentParams


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
    obs = jnp.expand_dims(obs, axis=0)
    return obs


def test_dqn_agent_act(dqn_agent, obs):
    ag_params = DQNAgentParams()
    env_params = DroneEnvParams()
    rng = jax.random.PRNGKey(0)
    ag_state = dqn_agent.reset(rng, ag_params, env_params)
    action = dqn_agent.act(rng, obs, ag_state)
    assert action.shape == ()
    assert action.dtype == jnp.int32

def test_dqn_agent_save(dqn_agent, obs):
    env_params = DroneEnvParams()
    ag_params = DQNAgentParams()
    rng = jax.random.PRNGKey(0)
    ag_state = dqn_agent.reset(rng, ag_params, env_params)
    out_before = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tf:
        # saving
        dqn_agent.save(tf.file.name, ag_state, ag_params, env_params)
        # loading again
        ag_state = dqn_agent.load(tf.file.name, ag_state)
        # assert exception
        with pytest.raises(Exception):
            ag_state = dqn_agent.load_from_torch(tf.file.name, ag_state)
    out_after = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
    assert jnp.allclose(out_before, out_after)

@pytest.mark.parametrize("hidden_layers", [
    (),
    (16, 64),
    ((8, 8, 8)),
])
def test_dqn_dense_agent_torch_save(hidden_layers, dqn_agent, obs):
    env_params = DroneEnvParams()
    ag_params = DQNAgentParams(network_type='dense', hidden_layers=hidden_layers)
    rng = jax.random.PRNGKey(1)
    ag_state = dqn_agent.reset(rng, ag_params, env_params)
    out_before = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tf:
        # saving
        dqn_agent.save_as_torch(tf.file.name, ag_state, ag_params, env_params)
        # loading again
        ag_state = dqn_agent.load_from_torch(tf.file.name, ag_state)
        # assert exception
        with pytest.raises(Exception):
            ag_state = dqn_agent.load(tf.file.name, ag_state)
    out_after = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
    assert jnp.allclose(out_before, out_after)

def test_dqn_conv_agent_act(dqn_agent, obs):
    env_params = DroneEnvParams()
    ag_params = DQNAgentParams(network_type='conv')
    rng = jax.random.PRNGKey(0)
    ag_state = dqn_agent.reset(rng, ag_params, env_params)
    out_before = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tf:
        # saving
        dqn_agent.save(tf.file.name, ag_state, ag_params, env_params)
        # loading again
        ag_state = dqn_agent.load(tf.file.name, ag_state)
        # assert exception
        with pytest.raises(Exception):
            ag_state = dqn_agent.load_from_torch(tf.file.name, ag_state)
    out_after = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
    assert jnp.allclose(out_before, out_after)

@pytest.mark.parametrize("conv_layers,conv_dense_layers", [
    (({'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},), ()),
    (({'out_channels': 32, 'kernel_size': 1, 'stride': 1, 'padding': 1},), (16, 8)),
    (({'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}, {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}), (16, 8)),
    ((), (8, 8, 8)),
])
def test_dqn_agent_conv_torch_save(conv_layers, conv_dense_layers, dqn_agent, obs):
    env_params = DroneEnvParams()
    ag_params = DQNAgentParams(network_type='conv', conv_layers=conv_layers, conv_dense_layers=conv_dense_layers)
    rng = jax.random.PRNGKey(1)
    ag_state = dqn_agent.reset(rng, ag_params, env_params)
    out_before = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tf:
        # saving
        dqn_agent.save_as_torch(tf.file.name, ag_state, ag_params, env_params)
        # loading again
        ag_state = dqn_agent.load_from_torch(tf.file.name, ag_state)
        # assert exception
        with pytest.raises(Exception):
            ag_state = dqn_agent.load(tf.file.name, ag_state)
    out_after = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
    assert jnp.allclose(out_before, out_after)
