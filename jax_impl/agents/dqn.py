from typing import Tuple, Dict, Union, Optional, Literal
import copy
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

from ..env.env import DroneEnvParams
from common.constants import Action



@dataclass
class DQNAgentParams:
    hidden_layers: Tuple[int, ...] = (32, 32)
    network_type: Literal['dense', 'conv'] = 'dense'
    conv_layers: Tuple[Dict[str, int], ...] = ({'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1},)
    conv_dense_layers: Tuple[int, ...] = ()
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.999
    epsilon_end: float = 0.01
    epsilon_decay_every: Optional[int] = None  # decay epsilon after every n steps of training. By default decay at episode end
    learning_rate: float = 1e-3
    target_update_interval: int = 5            # interval for updating target network
    tau: float = 1.0                           # EMA decay rate / smoothing coef for target network


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
        # x = x.reshape(x.shape[0], -1)
        for n_features in self.hidden_layers:
            x = nn.Dense(n_features, kernel_init=nn.initializers.he_normal())(x)
            x = nn.relu(x)
        x = nn.Dense(Action.num_actions())(x)
        return x

    def init_weights(
            self,
            rng: jnp.ndarray,
            input_shape: Tuple[int, int]):
        return self.init({'params': rng}, jnp.zeros(input_shape, dtype=jnp.float32))


class ConvQNetwork(nn.Module):
    obs_shape: Tuple[int, ...]
    conv_layers: Tuple[Dict[str, int], ...] = ({"out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},)
    dense_layers: Tuple[int, ...] = ()

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], *self.obs_shape)
        for conv_kwds in self.conv_layers:
            x = nn.Conv(
                features=conv_kwds["out_channels"],
                kernel_size=(conv_kwds["kernel_size"], conv_kwds["kernel_size"]),
                strides=(conv_kwds.get("stride", 1), conv_kwds.get("stride", 1)),
                padding=conv_kwds.get("padding", 0),
                )(x)
            x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)
        for n_features in self.dense_layers:
            x = nn.Dense(features=n_features)(x)
            x = nn.relu(x)
        x = nn.Dense(Action.num_actions())(x)
        return x

    def init_weights(
            self,
            rng: jnp.ndarray,
            input_shape: Tuple[int, int]):
        return self.init({'params': rng}, jnp.zeros(input_shape, dtype=jnp.float32))


class DQNAgent():
    def reset(self, rng: jnp.ndarray, ag_params: DQNAgentParams, env_params: DroneEnvParams) -> DQNAgentState:
        if env_params.wrapper != 'window':
            raise NotImplementedError
        # create network
        r = env_params.window_radius
        obs_shape = (r * 2 + 1, r * 2 + 1, 6)
        if ag_params.network_type == 'dense':
            qnetwork = DenseQNetwork(ag_params.hidden_layers)
        elif ag_params.network_type == 'conv':
            qnetwork = ConvQNetwork(
                    obs_shape=obs_shape,
                    conv_layers=ag_params.conv_layers,
                    dense_layers=ag_params.conv_dense_layers)
        else:
            raise ValueError(f'Unsupported network type {ag_params.network_type}')
        input_shape = (1, (env_params.window_radius * 2 + 1) ** 2 * 6)
        qnetwork_params = qnetwork.init_weights(rng, input_shape)
        # create optimizer
        optimizer = optax.adam(ag_params.learning_rate)
        opt_state = optimizer.init(qnetwork_params)
        # create target network (static)
        rng, key = jax.random.split(rng)
        target_qnetwork = copy.deepcopy(qnetwork)
        target_qnetwork_params = target_qnetwork.init_weights(key, input_shape)
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
            # q_values are (BS, 5), actions are (BS,) - we need (BS, 1) for take_along_axis to work
            q_value = jnp.take_along_axis(q_values, jnp.expand_dims(batch['actions'], axis=1), axis=1)
            q_value = jnp.squeeze(q_value)

            # Q-values for next state (target network)
            next_q_values = ag_state.target_qnetwork.apply(
                    ag_state.target_qnetwork_params,
                    batch['next_obs'])
            next_q_value = jnp.max(next_q_values, axis=1)

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

    def should_update_epsilon(self, ag_params: DQNAgentParams, step: jnp.ndarray, done: jnp.ndarray):
        if ag_params.epsilon_decay_every is None:
            return done
        else:
            return step % ag_params.epsilon_decay_every == 0

    def update_epsilon(self, ag_state: DQNAgentState, ag_params: DQNAgentParams):
        epsilon = jnp.maximum(ag_state.epsilon * ag_params.epsilon_decay, ag_params.epsilon_end)
        return ag_state.replace(epsilon=epsilon)

    def load(self, path: str, ag_state: DQNAgentState):
        metadata = safe_open(path, 'np').metadata()
        if metadata.get('checkpoint_format') != 'jax':
            raise Exception(f'The checkpoint under {path} is not compatible with JAX')
        params = load_file(path)
        params = unflatten_dict(params, sep='.')
        if metadata['network_type'] == 'dense':
            hidden_layers = ast.literal_eval(metadata['dense_layers'])
            qnetwork = DenseQNetwork(hidden_layers)
        elif metadata['network_type'] == 'conv':
            conv_layers = ast.literal_eval(metadata['conv_layers'])
            conv_dense_layers = ast.literal_eval(metadata['conv_dense_layers'])
            obs_shape = ast.literal_eval(metadata['obs_shape'])
            qnetwork = ConvQNetwork(
                    obs_shape=obs_shape,
                    conv_layers=conv_layers,
                    dense_layers=conv_dense_layers)
        else:
            raise ValueError(f'Unexpected network type {metadata["network_type"]}')
        ag_state = ag_state.replace(
                qnetwork=qnetwork,
                target_qnetwork=copy.deepcopy(qnetwork),
                qnetwork_params=params,
                target_qnetwork_params=params)
        return ag_state

    def load_from_torch(self, path: str, ag_state: DQNAgentState):
        metadata = safe_open(path, 'np').metadata()
        if metadata.get('checkpoint_format', 'torch') != 'torch':
            raise Exception(f'The checkpoint under {path} is not a PyTorch checkpoint')
        if metadata.get('network_type', 'dense') not in ['dense', 'conv']:
            raise Exception(f'The checkpoint under {path} is of network type {metadata.get("network_type")} which is currently not supported.')
        params = load_file(path)
        new_params = {}
        for original_key, v in params.items():
            key = original_key.split('.')
            if key[0] == 'network':
                key[0] = 'params'
            if key[1].startswith(('dense', 'conv')):
                new_key_name = key[1].capitalize()  # dense => Dense / conv => Conv
                new_key_name, layer_idx = new_key_name.split('_')
                new_key_name = new_key_name + '_' + str(int(layer_idx) - 1)
                key[1] = new_key_name
            if key[-1] == 'weight':
                if key[1].startswith('Dense'):
                    v = v.T
                elif key[1].startswith('Conv'):
                    v = v.transpose((2, 3, 1, 0))
                else:
                    raise Exception(f'Unexpected key {key}')
                key[-1] = 'kernel'
            new_key = '.'.join(key)
            new_params[new_key] = v
        params = new_params
        params = unflatten_dict(params, sep='.')
        if metadata['network_type'] == 'dense':
            hidden_layers = ast.literal_eval(metadata['dense_layers'])
            qnetwork = DenseQNetwork(hidden_layers)
        elif metadata['network_type'] == 'conv':
            conv_layers = ast.literal_eval(metadata['conv_layers'])
            conv_dense_layers = ast.literal_eval(metadata['conv_dense_layers'])
            obs_shape = ast.literal_eval(metadata['obs_shape'])
            qnetwork = ConvQNetwork(
                    obs_shape=obs_shape,
                    conv_layers=conv_layers,
                    dense_layers=conv_dense_layers)
        else:
            raise ValueError(f'Unexpected network type {metadata["network_type"]}')
        ag_state = ag_state.replace(
                qnetwork=qnetwork,
                target_qnetwork=copy.deepcopy(qnetwork),
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
            "network_type": ag_params.network_type,
            "dense_layers": str(ag_params.hidden_layers),
            "conv_layers": str(ag_params.conv_layers),
            "conv_dense_layers": str(ag_params.conv_dense_layers),
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
            "network_type": ag_params.network_type,
            "dense_layers": str(ag_params.hidden_layers),
            "conv_dense_layers": str(ag_params.conv_dense_layers),
            "conv_layers": str(ag_params.conv_layers),
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
            original_shape = v.shape
            if key[0] == 'params':
                key[0] = 'network'
            if key[1].startswith(('Dense', 'Conv')):
                new_key_name = key[1].lower()  # Dense => dense / Conv => conv
                new_key_name, layer_idx = new_key_name.split('_')
                new_key_name = new_key_name + '_' + str(int(layer_idx) + 1)
                key[1] = new_key_name
            if key[-1] == 'kernel':
                if key[1].startswith('dense'):
                    v = v.T
                elif key[1].startswith('conv'):
                    v = jnp.transpose(v, (3, 2, 0, 1))
                else:
                    raise Exception(f'Unexpected key {key}')
                key[-1] = 'weight'
            new_key = '.'.join(key)
            # print(f'Saving {original_key} => {new_key} (shape: {v.shape}, original shape: {original_shape})')
            new_params[new_key] = v
        params = new_params
        save_file(params, save_path, metadata=metadata)
