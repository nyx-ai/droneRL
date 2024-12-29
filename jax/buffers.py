from typing import Dict
import jax.numpy as jnp
import jax.random

from flax.struct import dataclass



@dataclass
class BufferState:
    experiences: Dict[str, jnp.ndarray]
    buffer_size: jnp.ndarray
    current_idx: jnp.ndarray
    current_size: jnp.ndarray


class ReplayBuffer:
    def init(
            self,
            buffer_size: int,
            experience: Dict[str, jnp.ndarray]
            ):
        experiences = jax.tree.map(jnp.empty_like, experience)
        experiences = jax.tree.map(lambda x: jnp.broadcast_to(x, (buffer_size, *x.shape)), experiences)
        state = BufferState(
                experiences=experiences,
                current_idx=jnp.array(0, dtype=jnp.int32),
                current_size=jnp.array(0, dtype=jnp.int32),
                buffer_size=jnp.array(buffer_size, dtype=jnp.int32)
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
                current_idx=(state.current_idx + 1) % state.buffer_size,
                current_size=jnp.minimum(state.current_size + 1, state.buffer_size),
                )
        return state

    def sample(self, rng: jnp.ndarray, batch_size: int, state: BufferState):
        rng, sampling_rng = jax.random.split(rng)
        indices = jax.random.randint(
            sampling_rng,
            shape=(batch_size,),
            minval=0,
            maxval=state.current_size
        )
        batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, indices, axis=0),
                state.experiences,
                )
        return batch, rng
