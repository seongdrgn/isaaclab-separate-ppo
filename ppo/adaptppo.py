from typing import Any, Mapping, Optional, Tuple, Union

import copy
import datetime
import os
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
# from skrl.agents.torch import Agent
from ppo.base import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

from models.model import AdaptationModule

# fmt: off
# [start-config-dict-torch]
AdaptPPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "observation_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "observation_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.state_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "adaptation_module": {
        "seq_len": 20,
        "input_dim": 32,          # input dimension of the adaptation module
        "heads": 2,              # number of attention heads in the transformer
        "layers": 2,             # number of transformer layers in the adaptation module
        "model_dim": 32,         # model dimension of the transformer in the adaptation module
    },

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class AdaptPPO(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        state_space: Optional[gymnasium.Space] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Proximal Policy Optimization (PPO).

        https://arxiv.org/abs/1707.06347

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        _cfg = copy.deepcopy(AdaptPPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        self.privileged_net = self.models.get("privileged_net", None)

        visuo_prop_sequence_length = _cfg["adaptation_module"]["seq_len"]
        self.seq_len = visuo_prop_sequence_length
        adaptation_transformer_input_dim = _cfg["adaptation_module"]["input_dim"]
        adaptation_transformer_heads = _cfg["adaptation_module"]["heads"]
        adaptation_transformer_layers = _cfg["adaptation_module"]["layers"]
        adaptation_transformer_model_dim = _cfg["adaptation_module"]["model_dim"]
        self.adaptation_module = AdaptationModule(visuo_prop_transformer_input_dim=adaptation_transformer_input_dim, 
                                            visuo_prop_transformer_heads=adaptation_transformer_heads, 
                                            visuo_prop_transformer_layers=adaptation_transformer_layers, 
                                            visuo_prop_transformer_model_dim=adaptation_transformer_model_dim,
                                            visuo_prop_sequence_length=visuo_prop_sequence_length,
                                            device=self.device)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        self.checkpoint_modules["privileged_net"] = self.privileged_net

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._observation_preprocessor = self.cfg["observation_preprocessor"]
        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        self.optimizer = torch.optim.Adam(self.adaptation_module.parameters(), lr=self._learning_rate)
        # self.checkpoint_modules["optimizer"] = self.optimizer
        # if self.policy is not None and self.value is not None:
        #     if self.policy is self.value:
        #         self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        #     else:
        #         self.optimizer = torch.optim.Adam(
        #             itertools.chain(self.policy.parameters(), self.value.parameters(), self.privileged_net.parameters()), lr=self._learning_rate
        #         )
        #     if self._learning_rate_scheduler is not None:
        #         self.scheduler = self._learning_rate_scheduler(
        #             self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
        #         )

            # self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        # - observations
        if self._observation_preprocessor:
            self._observation_preprocessor = self._observation_preprocessor(
                **self.cfg["observation_preprocessor_kwargs"]
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        # - values
        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)

        # create tensors in memory

        if self.memory is not None:
            self.memory.create_tensor(name="images", size=(64*64), dtype=torch.float32)
            self.memory.create_tensor(name="input_sequence", size=(64*self.seq_len), dtype=torch.float32)
            self.memory.create_tensor(name="gt_action", size=(16), dtype=torch.float32)
            self.memory.create_tensor(name="observations", size=self.observation_space.shape, dtype=torch.float32)

            # self._tensors_names = ["privileged_features", "input_sequence", "prop_action","gt_action","observations"]
            self._tensors_names = ["images", "input_sequence","gt_action","observations"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None

    def act(self, observations: torch.Tensor, states: Union[torch.Tensor, None], prev_input_sequence: torch.Tensor, images: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        # get proprioceptive observations & previous actions
        prop = inputs["observations"][:,22*2+6:22*2+22]
        prev_actions = inputs["observations"][:,22*3+16*2:22*3+16*3]
        prop_actions = torch.cat((prop, prev_actions), dim=-1)

        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:           
            adapt_features, cur_input_sequence = self.adaptation_module(prev_input_sequence=prev_input_sequence, image=images, prop_action=prop_actions)
            actions = self.policy.random_act({"obs": inputs["observations"], "privileged_features": adapt_features}, role="policy")
            return actions, cur_input_sequence

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):            
            adapt_features, cur_input_sequence = self.adaptation_module(prev_input_sequence=prev_input_sequence, image=images, prop_action=prop_actions)
            actions, log_prob, outputs = self.policy.act({"obs": inputs["observations"], "privileged_features": adapt_features}, role="policy")
            self._current_log_prob = log_prob

        return actions, log_prob, cur_input_sequence, outputs

    def record_transition(
            self,
            *,
            images: torch.Tensor,
            input_sequence: torch.Tensor,
            gt_action: torch.Tensor,
            observations: torch.Tensor,
    ) -> None:
        """Record an environment transition in memory.

        :param privileged_features: Privileged features extracted from the environment.
        :param adapted_features: Adapted features for the agent.
        """
        if self.memory is not None:
            # store privileged and adapted features in memory
            self.memory.add_samples(
                images=images,
                input_sequence=input_sequence,
                gt_action=gt_action,
                observations=observations
            )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.enable_training_mode(True)
            self.update(timestep=timestep, timesteps=timesteps)
            self.enable_training_mode(False)

        # write tracking data and checkpoints
        # super().post_interaction(timestep=timestep, timesteps=timesteps)
        timestep += 1

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            self.write_checkpoint(timestep=timestep, timesteps=timesteps)

        # write to tensorboard
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep=timestep, timesteps=timesteps)

    def write_checkpoint(self, *, timestep: int, timesteps: int) -> None:
        """Write checkpoint (modules) to persistent storage.

        .. note::

            The checkpoints are stored in the subdirectory ``checkpoints`` within the experiment directory.
            The checkpoint name is the ``timestep`` argument value (if it is not ``None``),
            or the current system date-time otherwise.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        # separated modules
        name = "adaptation_module"
        torch.save(
            self._get_internal_value(self.adaptation_module),
            os.path.join(self.experiment_dir, "checkpoints", f"{name}_{tag}.pt"),
        )

    def update(self, *, timestep: int, timesteps: int) -> None:
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            for (
                sampled_images,
                sampled_prev_input_sequence,
                sampled_gt_action,
                sampled_observation
            ) in sampled_batches:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observation, train=not epoch),
                    }

                    # reshape tensors
                    sampled_images = sampled_images.reshape(-1, 64, 64, 1)
                    sampled_prev_input_sequence = sampled_prev_input_sequence.reshape(-1, self.seq_len, 64)
                    
                    # compute privileged features
                    privileged_states = inputs["observations"][:,-self.cfg["privileged_input_dim"]:]
                    privileged_features = self.privileged_net(privileged_states)
                    privileged_features = torch.tanh(privileged_features)

                    # get proprioceptive observations & previous actions
                    prop = inputs["observations"][:,22*2+6:22*2+22]
                    prev_actions = inputs["observations"][:,22*3+16*2:22*3+16*3]
                    prop_actions = torch.cat((prop, prev_actions), dim=-1)

                    # compute adapted features
                    adapted_features, _ = self.adaptation_module(prev_input_sequence=sampled_prev_input_sequence, image=sampled_images, prop_action=prop_actions)

                    # output estimated actions
                    actions = self.policy.act({"obs": inputs["observations"], "privileged_features": adapted_features}, role="policy")[0]
                    clamped_actions = torch.clamp(actions, -1.0, 1.0)
                    action_loss = F.mse_loss(sampled_gt_action, clamped_actions)

                    # post-process adapted features
                    adapted_features = torch.tanh(adapted_features)
                    
                    # compute adaptation loss
                    adaptation_loss = F.mse_loss(privileged_features, adapted_features)

                    # concatenate losses
                    loss = adaptation_loss
                    # loss = action_loss + adaptation_loss
                
                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_loss += loss.item()
        
        # record data
        self.track_data("Loss / Adaptation loss", cumulative_loss / (self._learning_epochs * self._mini_batches))