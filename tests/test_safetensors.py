import tempfile
import os
import pytest

from env.v3.env import DeliveryDrones
from env.v3.wrappers import WindowedGridView
from agents.dqn import DQNAgent, DenseQNetworkFactory, ConvQNetworkFactory


@pytest.mark.parametrize("factory_class,factory_params", [
    (DenseQNetworkFactory, {"hidden_layers": [32]}),
    (DenseQNetworkFactory, {"hidden_layers": [8, 8, 8]}),
    (ConvQNetworkFactory, {"conv_layers": [
     {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}], "dense_layers": [32]}),
    (ConvQNetworkFactory, {"conv_layers": [
     {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1}], "dense_layers": [32]}),
    (ConvQNetworkFactory, {"conv_layers": [
     {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1}], "dense_layers": [8, 8, 8]}),
])
def test_agent_save_load(factory_class, factory_params):
    # Create small environment
    env_params = {
        'n_drones': 1,
        'drone_density': 0.5,
        'packets_factor': 1,
        'dropzones_factor': 1,
        'stations_factor': 1,
        'skyscrapers_factor': 1
    }
    env = WindowedGridView(DeliveryDrones(env_params), radius=2)

    # Create agent
    factory = factory_class(env, **factory_params)
    agent = DQNAgent(
        env=env,
        dqn_factory=factory,
        gamma=0.99,
        epsilon_start=0,  # No exploration
        epsilon_decay=1,
        epsilon_end=0,
        memory_size=1000,
        batch_size=32,
        target_update_interval=100
    )

    # Get some states and actions
    states = []
    actions = []
    state = env.reset()
    for _ in range(10):
        states.append(state[0])  # Only one drone
        action = agent.act(state[0])
        actions.append(action)
        state, _, _, _, _ = env.step({0: action})

    # Save agent
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        agent.save(f.name)
        saved_path = f.name

    # Create new agent and load weights
    new_agent = DQNAgent(
        env=env,
        dqn_factory=factory,
        gamma=0.99,
        epsilon_start=0,
        epsilon_decay=1,
        epsilon_end=0,
        memory_size=1000,
        batch_size=32,
        target_update_interval=100
    )
    new_agent.load(saved_path)

    # Verify actions are the same
    new_actions = []
    for state in states:
        new_actions.append(new_agent.act(state))

    os.unlink(saved_path)

    assert actions == new_actions, "Actions before and after loading should be identical"
    print("Actions before and after loading are identical")


if __name__ == "__main__":
    pytest.main([__file__])
