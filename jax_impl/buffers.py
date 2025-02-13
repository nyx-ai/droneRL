from typing import Dict
import jax.numpy as jnp
import jax.random

from flax.struct import dataclass



@dataclass
class BufferState:
    experiences: Dict[str, jnp.ndarray]
    current_idx: jnp.ndarray
    current_size: jnp.ndarray


class ReplayBuffer:
    sample_batch_size: int = 64
    buffer_size: int = 10000

    def __init__(self, buffer_size: int = 10_000, sample_batch_size: int = 64):
        self.buffer_size = buffer_size
        self.sample_batch_size = sample_batch_size

    def init(
            self,
            experience: Dict[str, jnp.ndarray]
            ):
        experiences = jax.tree.map(jnp.empty_like, experience)
        experiences = jax.tree.map(lambda x: jnp.broadcast_to(x, (self.buffer_size, *x.shape)), experiences)
        state = BufferState(
                experiences=experiences,
                current_idx=jnp.array(0, dtype=jnp.int32),
                current_size=jnp.array(0, dtype=jnp.int32),
                )
        return state

    def add(
            self,
            state: BufferState,
            experience: Dict[str, jnp.ndarray]
            ):
        experiences = jax.tree_util.tree_map(
                lambda experience_field, batch_field: experience_field.at[state.current_idx].set(
                    batch_field
                    ),
                state.experiences,
                experience,
                )
        state = state.replace(
                experiences=experiences,
                current_idx=(state.current_idx + 1) % self.buffer_size,
                current_size=jnp.minimum(state.current_size + 1, self.buffer_size),
                )
        return state

    def add_many(
            self,
            state: BufferState,
            experiences: Dict[str, jnp.ndarray]
            ):
        num_experiences = jax.tree_util.tree_leaves(experiences)[0].shape[0]
        indices = (state.current_idx + jnp.arange(num_experiences)) % self.buffer_size

        experiences = jax.tree_util.tree_map(
                lambda experience_field, batch_field: experience_field.at[indices].set(
                    batch_field
                    ),
                state.experiences,
                experiences,
                )
        state = state.replace(
                experiences=experiences,
                current_idx=(state.current_idx + num_experiences) % self.buffer_size,
                current_size=jnp.minimum(state.current_size + num_experiences, self.buffer_size),
                )
        return state

    def sample(self, key: jnp.ndarray, state: BufferState):
        indices = jax.random.randint(
            key,
            shape=(self.sample_batch_size,),
            minval=0,
            maxval=state.current_size
        )
        batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, indices, axis=0),
                state.experiences,
                )
        return batch


    def can_sample(self, state: BufferState):
        return state.current_size >= self.sample_batch_size
