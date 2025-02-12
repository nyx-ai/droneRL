from typing import Tuple, Dict, Union, Literal
import ast
import jax
import jax.random
import jax.numpy as jnp
from flax.struct import dataclass, field
from flax.core import FrozenDict
from flax import linen as nn
import optax
from safetensors.numpy import save_file, load_file
from safetensors import safe_open
from flax.traverse_util import flatten_dict, unflatten_dict

# from jax_impl.env.env import DroneEnvParams
# from jax_impl.env.constants import Action
from ..env.env import DroneEnvParams
from ..env.constants import Action
# from env.env import DroneEnvParams
# from env.constants import Action



@dataclass
class DQNAgentParams:
    batch_size: int = 1
    hidden_layers: Tuple[int, ...] = (32, 32)
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.999
    epsilon_end: float = 0.01
    learning_rate: float = 1e-3
    target_update_interval: int = 5       # interval for updating target network
    tau: float = 1.0                      # EMA decay rate / smoothing coef for target network


@dataclass
class DQNAgentState:
    qnetwork: nn.Module = field(pytree_node=False)
    target_qnetwork: nn.Module = field(pytree_node=False)
    qnetwork_params: Union[Dict, FrozenDict]
    target_qnetwork_params: Union[Dict, FrozenDict]
    optimizer: optax.GradientTransformation = field(pytree_node=False)
    opt_state: optax.OptState
    epsilon: jnp.ndarray


class DenseQNetwork(nn.Module):
    hidden_layers: Tuple[int, ...] = (32, 32)

    @nn.compact
    def __call__(self, x):
        for n_features in self.hidden_layers:
            x = nn.Dense(n_features)(x)
            x = nn.relu(x)
        x = nn.Dense(Action.num_actions())(x)
        return x


class DQNAgent():
    def reset(self, rng: jnp.ndarray, ag_params: DQNAgentParams, env_params: DroneEnvParams) -> DQNAgentState:
        if env_params.wrapper != 'window':
            raise NotImplementedError
        r = env_params.window_radius
        input_size = (r * 2 + 1) ** 2 * 6
        # create network
        qnetwork = DenseQNetwork(ag_params.hidden_layers)
        qnetwork_params = qnetwork.init({'params': rng}, jnp.zeros((1, input_size)))
        # create optimizer
        optimizer = optax.adam(ag_params.learning_rate)
        opt_state = optimizer.init(qnetwork_params)
        # create target network (static)
        rng, key = jax.random.split(rng)
        target_qnetwork = DenseQNetwork(ag_params.hidden_layers)
        target_qnetwork_params = qnetwork.init({'params': key}, jnp.zeros((1, input_size)))
        return DQNAgentState(
                qnetwork=qnetwork,
                qnetwork_params=qnetwork_params,
                target_qnetwork=target_qnetwork,
                target_qnetwork_params=target_qnetwork_params,
                optimizer=optimizer,
                opt_state=opt_state,
                epsilon=jnp.array(ag_params.epsilon_start)
                )

    def act(self, key: jnp.ndarray, obs: jnp.ndarray, ag_state: DQNAgentState, greedy: bool = False):

        def _explore():
            return jax.random.randint(key, shape=(), minval=0, maxval=Action.num_actions())

        def _exploit():
            out = ag_state.qnetwork.apply(ag_state.qnetwork_params, obs)
            return jnp.argmax(out)

        if greedy:
            return _exploit()

        rand_val = jax.random.uniform(key)
        return jax.lax.cond(rand_val < ag_state.epsilon, _explore, _exploit)

    def train_step(
            self,
            ag_state: DQNAgentState,
            batch: Dict[str, jnp.ndarray],
            ag_params: DQNAgentParams):

        def compute_loss(network_params):
            # Q-values for current state (network)
            q_values = ag_state.qnetwork.apply(network_params, batch['obs'])
            q_value = jnp.take(q_values, batch['actions'])

            # Q-values for next state (target network)
            next_q_values = ag_state.target_qnetwork.apply(
                    ag_state.target_qnetwork_params,
                    batch['next_obs'])
            next_q_value = jnp.max(next_q_values)

            # Bellman equation
            td_target = batch['rewards'] + ag_params.gamma * next_q_value * (1 - batch['dones'])

            # MSE loss
            loss = jnp.mean(jnp.square(q_value - td_target))
            return loss

        # compute gradient
        loss, grads = jax.value_and_grad(compute_loss)(ag_state.qnetwork_params)

        # compute optimizer & weight updates
        updates, new_opt_state = ag_state.optimizer.update(
                grads,
                ag_state.opt_state,
                ag_state.qnetwork_params)
        new_qnetwork_params = optax.apply_updates(ag_state.qnetwork_params, updates)
        ag_state = ag_state.replace(qnetwork_params=new_qnetwork_params, opt_state=new_opt_state)
        return ag_state, loss

    def update_target(self, ag_state: DQNAgentState, ag_params: DQNAgentParams):
        target_params = optax.incremental_update(
            ag_state.qnetwork_params,
            ag_state.target_qnetwork_params,
            ag_params.tau)
        return ag_state.replace(target_qnetwork_params=target_params)

    def update_epsilon(self, ag_state: DQNAgentState, ag_params: DQNAgentParams):
        epsilon = jnp.maximum(ag_state.epsilon * ag_params.epsilon_decay, ag_params.epsilon_end)
        return ag_state.replace(epsilon=epsilon)

    def load(self, path: str, ag_state: DQNAgentState):
        metadata = safe_open(path, 'np').metadata()
        if metadata.get('checkpoint_format') != 'jax':
            raise Exception(f'The checkpoint under {path} is not compatible with JAX')
        params = load_file(path)
        params = unflatten_dict(params, sep='.')
        hidden_layers = ast.literal_eval(metadata['dense_layers'])
        qnetwork = DenseQNetwork(hidden_layers)
        ag_state = ag_state.replace(
                qnetwork=qnetwork,
                qnetwork_params=params,
                target_qnetwork_params=params)
        return ag_state

    def load_from_torch(self, path: str, ag_state: DQNAgentState):
        metadata = safe_open(path, 'np').metadata()
        if metadata.get('checkpoint_format', 'torch') != 'torch':
            raise Exception(f'The checkpoint under {path} is not a PyTorch checkpoint')
        if metadata.get('network_type', 'dense') not in ['dense']:
            raise Exception(f'The checkpoint under {path} is of network type {metadata.get("network_type")} which is currently not supported.')
        params = load_file(path)
        new_params = {}
        for original_key, v in params.items():
            key = original_key.split('.')
            if key[0] == 'network':
                key[0] = 'params'
            if key[1].startswith('dense'):
                new_key_name = key[1].capitalize()  # dense => Dense
                new_key_name, layer_idx = new_key_name.split('_')
                new_key_name = new_key_name + '_' + str(int(layer_idx) - 1)
                key[1] = new_key_name
            if key[-1] == 'weight':
                v = v.T
                key[-1] = 'kernel'
            new_key = '.'.join(key)
            new_params[new_key] = v
        params = new_params
        params = unflatten_dict(params, sep='.')
        hidden_layers = ast.literal_eval(metadata.get('dense_layers'))
        qnetwork = DenseQNetwork(hidden_layers)
        ag_state = ag_state.replace(
                qnetwork=qnetwork,
                qnetwork_params=params,
                target_qnetwork_params=params)
        return ag_state

    def save(
            self,
            save_path: str,
            ag_state: DQNAgentState,
            ag_params: DQNAgentParams,
            env_params: DroneEnvParams,
            checkpoint_format_version: float = 0.1,
            ):
        window_size = env_params.window_radius * 2 + 1
        metadata = {
            "network_type": "dense",
            "dense_layers": str(ag_params.hidden_layers),
            "obs_shape": str((window_size, window_size, 6)),
            "action_shape": str((Action.num_actions(),)),
            "checkpoint_format": 'jax',
            "checkpoint_format_version": str(checkpoint_format_version),
        }
        params = jax.device_get(ag_state.qnetwork_params)
        params = dict(flatten_dict(params, sep='.'))
        save_file(params, save_path, metadata=metadata)

    def save_as_torch(
            self,
            save_path: str,
            ag_state: DQNAgentState,
            ag_params: DQNAgentParams,
            env_params: DroneEnvParams,
            checkpoint_format_version: float = 0.1,
            ):
        window_size = env_params.window_radius * 2 + 1
        metadata = {
            "network_type": "dense",
            "dense_layers": str(ag_params.hidden_layers),
            "obs_shape": str((window_size, window_size, 6)),
            "action_shape": str((Action.num_actions(),)),
            "checkpoint_format": 'torch',
            "checkpoint_format_version": str(checkpoint_format_version),
        }
        params = jax.device_get(ag_state.qnetwork_params)
        params = dict(flatten_dict(params, sep='.'))
        # some renaming to make the checkpoint compatible with the PyTorch implementation
        new_params = {}
        for original_key, v in params.items():
            key = original_key.split('.')
            if key[0] == 'params':
                key[0] = 'network'
            if key[1].startswith('Dense'):
                new_key_name = key[1].lower()  # Dense => dense
                new_key_name, layer_idx = new_key_name.split('_')
                new_key_name = new_key_name + '_' + str(int(layer_idx) + 1)
                key[1] = new_key_name
            if key[-1] == 'kernel':
                v = v.T
                key[-1] = 'weight'
            new_key = '.'.join(key)
            new_params[new_key] = v
        params = new_params
        save_file(params, save_path, metadata=metadata)
