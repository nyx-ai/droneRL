import jax
import jax.numpy as jnp

NUM_ACTIONS = 5

class RandomAgent():
    """Random agent"""

    def __init__(self):
        pass

    def act(self, key):
        return jax.random.randint(key, (), 0, NUM_ACTIONS, dtype=jnp.int32)

    def reset(self):
        pass

    def learn(self, state, action, reward, next_state, done):
        pass
