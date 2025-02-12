import jax
import os
import tempfile

from jax_impl.env.env import DroneEnvParams
from jax_impl.agents.dqn import DQNAgent, DQNAgentParams
from jax_impl.render_util import render_video


def test_render_video():
    env_params = DroneEnvParams(n_drones=3, grid_size=10, window_radius=2)
    dqn_agent = DQNAgent()
    rng = jax.random.PRNGKey(0)
    ag_params = DQNAgentParams()
    ag_state = dqn_agent.reset(rng, ag_params, env_params)
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tf:
        mp4_out = tf.file.name + '.mp4'
        render_video(env_params, ag_state, output_path=mp4_out, num_steps=3)
        assert os.path.isfile(mp4_out)
