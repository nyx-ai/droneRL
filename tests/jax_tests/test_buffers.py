import pytest
import jax.numpy as jnp
import jax.random
from jax_impl.buffers import ReplayBuffer, BufferState

@pytest.fixture
def experience_example():
    """Creates a sample experience dictionary."""
    return {
        "state": jnp.array([1.0, 2.0, 3.0]),
        "action": jnp.array([0]),
        "reward": jnp.array([1.0]),
        "next_state": jnp.array([4.0, 5.0, 6.0]),
        "done": jnp.array([0])
    }

@pytest.fixture
def buffer():
    """Creates a ReplayBuffer instance."""
    return ReplayBuffer(buffer_size=100, sample_batch_size=10)

@pytest.fixture
def buffer_state(buffer, experience_example):
    """Initializes a buffer state."""
    return buffer.init(experience_example)

@pytest.fixture
def filled_buffer_state(buffer, buffer_state, experience_example):
    """Fills the buffer with enough experiences to allow sampling."""
    state = buffer_state
    for _ in range(15):  # Fill with 15 experiences
        state = buffer.add(state, experience_example)
    return state


def test_buffer_initialization(buffer, buffer_state):
    """Tests whether the buffer initializes correctly."""
    assert isinstance(buffer_state, BufferState)
    assert buffer_state.current_idx == 0
    assert buffer_state.current_size == 0
    for key, value in buffer_state.experiences.items():
        assert value.shape[0] == buffer.buffer_size


def test_add_experience(buffer, buffer_state, experience_example):
    """Tests adding an experience to the buffer."""
    new_state = buffer.add(buffer_state, experience_example)
    assert new_state.current_idx == 1
    assert new_state.current_size == 1
    for key in experience_example:
        assert jnp.all(new_state.experiences[key][0] == experience_example[key])


def test_buffer_wrapping(buffer, buffer_state, experience_example):
    """Tests buffer overwriting behavior when full."""
    state = buffer_state
    for i in range(buffer.buffer_size + 5):  # Overflow the buffer
        modified_experience = {k: v + i for k, v in experience_example.items()}
        state = buffer.add(state, modified_experience)

    assert state.current_idx == 5  # Because of circular overwriting
    assert state.current_size == buffer.buffer_size


def test_sample(buffer, filled_buffer_state):
    """Tests whether sampling returns the correct shape."""
    key = jax.random.PRNGKey(0)
    batch = buffer.sample(key, filled_buffer_state)
    assert isinstance(batch, dict)
    for key, value in batch.items():
        assert value.shape[0] == buffer.sample_batch_size


def test_can_sample(buffer, filled_buffer_state, buffer_state):
    """Tests whether the can_sample function works correctly."""
    assert buffer.can_sample(filled_buffer_state)
    assert not buffer.can_sample(buffer_state)


# @pytest.mark.focus
def test_add_many_experiences(buffer, buffer_state, experience_example):
    """Tests adding multiple experiences at once to the buffer."""
    # Create a batch of experiences
    batch_size = 5
    batch_experiences = {
        key: jnp.stack([value + i for i in range(batch_size)])
        for key, value in experience_example.items()
    }

    # Add the batch to the buffer
    new_state = buffer.add_many(buffer_state, batch_experiences)

    # Test that indices and size updated correctly
    assert new_state.current_idx == batch_size
    assert new_state.current_size == batch_size

    # Verify all experiences were stored correctly
    for i in range(batch_size):
        for key in experience_example:
            assert jnp.all(new_state.experiences[key][i] == experience_example[key] + i)

    # Test wrapping behavior
    # Create a batch that will wrap around the buffer
    wrap_size = 3
    start_idx = buffer.buffer_size - wrap_size
    wrap_state = buffer_state.replace(current_idx=jnp.array(start_idx))

    wrap_experiences = {
        key: jnp.stack([value + i for i in range(batch_size)])
        for key, value in experience_example.items()
    }

    wrapped_state = buffer.add_many(wrap_state, wrap_experiences)

    # Check that index wrapped correctly
    expected_idx = (start_idx + batch_size) % buffer.buffer_size
    assert wrapped_state.current_idx == expected_idx

    # Verify experiences at the end of buffer
    for i in range(wrap_size):
        idx = start_idx + i
        for key in experience_example:
            assert jnp.all(wrapped_state.experiences[key][idx] == experience_example[key] + i)

    # Verify wrapped experiences at start of buffer
    for i in range(batch_size - wrap_size):
        for key in experience_example:
            assert jnp.all(wrapped_state.experiences[key][i] == experience_example[key] + (wrap_size + i))
