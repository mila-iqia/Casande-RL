import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
import torch
from torch import nn
import numpy as np

import os
from abc import ABC
from copy import deepcopy

from QMR import QMR


def run_ray(args):
    ray.init()

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    ray.tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": args.training_iteration},
        config={
            "env": QMR,
            "env_config": {"args": deepcopy(args)},
            "lr": 1e-5,
            "gamma": args.gamma,
            "model": {
                "custom_model": "dense_model",
            },
            "num_workers": os.cpu_count() - 1,
            "num_gpus": torch.cuda.device_count(),
            "framework": "torch",
            "evaluation_interval": args.training_iteration,
            "evaluation_num_episodes": args.test_size,
            "evaluation_num_workers": os.cpu_count() - 1,
        },
        checkpoint_at_end=True,
        resume=args.resume
    )
    ray.shutdown()
# class Agent:
#     def __init__(self, args):
#         ray.init()
#         ModelCatalog.register_custom_model("ac_model", ActorCriticModel)
#         self.args = args

#     def run(self):
#         if self.args.algorithm == 'mcts':
#             algo = "contrib/AlphaZero"
#         else:
#             algo = self.args.algorithm

#         ray.tune.run(
#             algo,
#             stop={"training_iteration": self.args.training_iteration},
#             config={
#                 "env": QMR,
#                 "env_config": {"args": self.args},
#                 "lr": 1e-5,
#                 "gamma": self.args.gamma,
#                 "model": {
#                     "custom_model": "ac_model",
#                 },
#                 "num_workers": os.cpu_count() - 1,
#                 "num_gpus": torch.cuda.device_count(),
#                 "framework": "torch",
#                 "evaluation_interval": self.args.training_iteration,
#                 "evaluation_num_episodes": self.args.test_size,
#                 "evaluation_num_workers": os.cpu_count() - 1,
#             },
#             checkpoint_at_end=True,
#             resume=self.args.resume
#         )

#     def __del__(self):
#         ray.shutdown()

# class ActorCriticModel(TorchModelV2, nn.Module, ABC):
#     def __init__(self, obs_space, action_space, num_outputs, model_config,
#                  name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
#                               model_config, name)
#         nn.Module.__init__(self)

#         self.preprocessor = get_preprocessor(
#             obs_space.original_space)(obs_space.original_space)

#         self.shared_layers = nn.Sequential(
#             nn.Linear(
#                 in_features=obs_space.original_space["obs"].shape[0],
#                 out_features=256),
#             nn.Linear(in_features=256, out_features=256)
#         )
#         self.actor_layers = nn.Sequential(
#             nn.Linear(in_features=256, out_features=action_space.n))
#         self.critic_layers = nn.Sequential(
#             nn.Linear(in_features=256, out_features=1))

#         self._value_out = None

#     def forward(self, input_dict, state, seq_lens):
#         x = input_dict["obs"]
#         x = self.shared_layers(x)
#         # actor outputs
#         logits = self.actor_layers(x)

#         # compute value
#         self._value_out = self.critic_layers(x)
#         return logits, None

#     def value_function(self):
#         return self._value_out

#     def compute_priors_and_value(self, obs):
#         obs = torch.as_tensor(
#             [self.preprocessor.transform(obs)], dtype=torch.float32)
#         input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

#         with torch.no_grad():
#             model_out = self.forward(input_dict, None, [1])
#             logits, _ = model_out
#             value = self.value_function()
#             logits, value = torch.squeeze(logits), torch.squeeze(value)
#             priors = nn.Softmax(dim=-1)(logits)

#             priors = priors.cpu().numpy()
#             value = value.cpu().numpy()

#             return priors, value
