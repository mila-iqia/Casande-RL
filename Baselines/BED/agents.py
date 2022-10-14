from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models import ModelCatalog
import ray.rllib.agents.ppo as ppo
from ray.rllib.models.preprocessors import get_preprocessor

# from ray.tune.registry import register_env
# from ray.rllib.examples.env.parametric_actions_cartpole import \
#     ParametricActionsCartPole
import ray
from ray.tune import grid_search
import torch
from torch import nn
from gym.spaces import Box

from QMR import QMR, QMRWrapper
import os
from abc import ABC

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38


class RLAgent:
    def __init__(self, args):
        ray.init()
        self.args = args
        self.algo = args.algorithm
        self.fcnet_hiddens = [2048, 2048, 1024]
        if self.algo == "AlphaZero":
            assert args.mask_actions
            self.algo = "contrib/AlphaZero"
            ModelCatalog.register_custom_model(
                "dense_model", ActorCriticModel)
        else:
            ModelCatalog.register_custom_model(
                "dense_model", ParametricActionsModel)

    def run(self):
        if self.args.eval:
            self.eval()
        else:
            self.train()

    def train(self):
        env_class = QMRWrapper if self.algo == "contrib/AlphaZero" else QMR
        config = {
            "env": env_class,
            "env_config": {"args": self.args},
            # "lr": grid_search([1e-7, 1e-6]),
            "lr": 1e-6,
            "gamma": self.args.gamma,
            "model": {
                "fcnet_hiddens": self.fcnet_hiddens,
                "fcnet_activation": "relu",
            },
            "num_workers": os.cpu_count() - 1,
            "num_gpus": torch.cuda.device_count(),
            "framework": "torch",
            "evaluation_interval": self.args.training_iteration,
            "evaluation_num_episodes": self.args.test_size,
        }
        if self.args.mask_actions or self.algo == 'contrib/AlphaZero':
            config["model"]["custom_model"] = "dense_model"

        ray.tune.run(
            self.algo,
            stop={"training_iteration": self.args.training_iteration},
            config=config,
            checkpoint_at_end=True,
            local_dir=self.args.local_dir,
            # resume=args.resume
        )

    def eval(self):
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = 1
        config["env_config"] = {"args": self.args}
        config["framework"] = "torch"
        config["model"] = {
            "custom_model": "dense_model",
            "vf_share_layers": True,
            "fcnet_hiddens": self.fcnet_hiddens,
            "fcnet_activation": "relu",
        }
        agent = ppo.PPOTrainer(config=config, env=QMR)
        agent.restore(self.args.checkpoint_path)

        # agent = analysis.get_last_checkpoint(
        #     metric="episode_reward_mean", mode="max")
        env = QMR({"args": self.args})

        n_correct = 0
        total_steps = 0
        test_size = self.args.test_size
        for _ in range(test_size):
            done = False
            obs = env.reset()
            while not done:
                action = agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
                total_steps += 1
            if reward > 0:
                n_correct += 1

        print(
            f'accuracy: {n_correct/test_size}; average steps: {total_steps/test_size}')

    def __del__(self):
        ray.shutdown()


class ParametricActionsModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        TorchModelV2.__init__(self,
                              obs_space,
                              action_space,
                              num_outputs,
                              model_config,
                              name)
        nn.Module.__init__(self)

        self.fc_model = FullyConnectedNetwork(
            Box(-1, 1, shape=(num_outputs - 1, )),
            action_space,
            num_outputs,
            model_config,
            name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.fc_model({
            "obs": input_dict["obs"]["obs"]
            # "obs": input_dict["obs"]["cart"]
        })

        # action_logits = avail_actions * action_logits

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.fc_model.value_function()


class ActorCriticModel(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.preprocessor = get_preprocessor(
            obs_space.original_space)(obs_space.original_space)

        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features=obs_space.original_space["obs"].shape[0],
                out_features=2048),
            nn.Linear(in_features=2048, out_features=1024)
        )
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=1024, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1))

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)
        # compute value
        self._value_out = self.critic_layers(x)
        return logits, None

    def value_function(self):
        return self._value_out

    def compute_priors_and_value(self, obs):
        # obs = torch.as_tensor(
        #     [self.preprocessor.transform(obs)], dtype=torch.float32)
        # input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        input_dict = {"obs": torch.as_tensor(obs["obs"], dtype=torch.float32)}
        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value
