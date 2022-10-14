import os
import random
import string

import numpy as np
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.buffer import buffer_from_example, numpify_buffer, torchify_buffer


class CustomSerialEvalCollector(SerialEvalCollector):
    """Overwites collect_evaluation function to render evaluation patients."""

    def __init__(
        self,
        envs,
        agent,
        TrajInfoCls,
        max_T,
        max_trajectories=None,
        out_path="",
        max_generation=200,
        topk=5,
        sample_indices_flag=False,
        seed=None,
    ):
        """Initilizes the SerialEvalCollector and rendering specific variables.

        Parameters
        ----------
        envs: list
            list of environments that will be used to evaluate the agent.
        agent: object
            the trained agent that need to be evaluated.
        TrajInfoCls: SimPaTrajEvalInfo
            class that keeps track of the trajectory information of the agent
            during the interaction.
        max_T: int
            maximum number of interaction/questions that will be
            asked over the entire evaluation set.
        max_trajectories: int
            maximum number of trajectories to be evaluated.
            Default: None
        out_path: str
            path to folder where the generated trajectories are to be
            stored/saved.
            Default: ""
        max_generation: int
            number of trajectories per pathology to be generated.
            Default: 200
        topk: int
            number of pathologies to output while rendering the patients.
            Default: 5
        sample_indices_flag: boolean
            flag indicating whether we sample the indices of the patients
            to evaluate upfront.
            Default: False
        seed: int
            seed used to sample the patient indices.
            Default: None

        Returns
        -------
        None

        """
        super(CustomSerialEvalCollector, self).__init__(
            envs, agent, TrajInfoCls, max_T, max_trajectories=max_trajectories
        )
        self.out_path = out_path
        self.max_generation = max_generation
        self.topk = topk
        self.seed = seed
        self.sample_indices_flag = sample_indices_flag
        numIndices = (
            self.envs[0].get_num_indices()
            if hasattr(self.envs[0], "get_num_indices")
            else None
        )
        shallWeSample = (
            (numIndices is not None)
            and (max_trajectories is not None)
            and (max_trajectories <= numIndices)
            and self.sample_indices_flag
        )
        self.sampleIndices = None
        self.sampleIndiceIndex = None
        if shallWeSample:
            all_indices = list(range(numIndices))
            if max_trajectories < numIndices:
                random.seed(self.seed)
                self.sampleIndices = random.sample(all_indices, max_trajectories)
            else:
                self.sampleIndices = all_indices
            self.sampleIndiceIndex = 0

    def clean(self, str):
        """Replaces punctuation in a string with `_` character.

        Parameters
        ----------
        str: str
            input string in which punctuation is to be replaced.

        Returns
        -------
        processed_str: str
            input string with punctuations replaced by `_`.

        """
        exclude = set(string.punctuation)
        processed_str = "".join([ch if ch not in exclude else "_" for ch in str])

        return processed_str

    def generate_trajectory(self, curr_pathology, patho_counts, env, agent_info, b):
        """Renders patients that have finished interacting with the agent.

        This function calls the render function in the simulator and
        generate trajectory for an interaction in the environment.

        Parameters
        ----------
        curr_pathology: str
            the current ground truth pathology that the patient has.
        patho_counts: dict
            a dictionary keeping track of the number of trajectories generated for
            various pathologies.
        env: object
            patient/environment with which the agent finished interacting with.
        agent_info: AgentInfo
            contains information about the distribution over pathologies for the
            current interaction.
        b: int
            id of the environment whose interaction is to be generated.

        Returns
        -------
        None

        """
        if patho_counts.get(curr_pathology, 0) <= self.max_generation:
            if not os.path.exists(f"{self.out_path}/trajs/"):
                os.makedirs(f"{self.out_path}/trajs/")
            patho_predictions = getattr(agent_info[b], "dist_info", None)
            if patho_predictions is not None:
                patho_predictions = patho_predictions.numpy()
            env.render(
                mode="all",
                filename=f"{self.out_path}/trajs/{curr_pathology}.txt",
                patho_predictions=patho_predictions,
                num=self.topk,
            )
            patho_counts[curr_pathology] = patho_counts.get(curr_pathology, 0) + 1

    def collect_evaluation(self, itr):
        """Collects trajectories of agent interaction with simulator.

        This function acts as an interface between the agent and the patients
        and collects the resulting interactions/trajectories.

        Parameters
        ----------
        itr: int
            this is used by the rlypt to set the evaluation mode.

        Returns
        -------
        completed_traj_infos: list
            list containing information about the individual completed trajectories
            like the pathology distribution, reward, length of the interaction, etc.

        """
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        for env in self.envs:
            o = (
                env.reset()
                if self.sampleIndices is None
                else env.reset_with_index(self.sampleIndices[self.sampleIndiceIndex])
            )
            self.sampleIndiceIndex = (
                None
                if self.sampleIndices is None
                else (
                    self.sampleIndiceIndex + 1
                    if self.sampleIndiceIndex + 1 < len(self.sampleIndices)
                    else 0  # re-iterate
                )
            )
            observations.append(o)
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(
            self.envs[0].action_space.null_value(), len(self.envs)
        )
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        patho_counts = {}
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(
                    observation[b], action[b], r, d, agent_info[b], env_info
                )
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    curr_pathology = self.clean(env.target_pathology)
                    self.generate_trajectory(
                        curr_pathology, patho_counts, env, agent_info, b
                    )
                    o = (
                        env.reset()
                        if self.sampleIndices is None
                        else env.reset_with_index(
                            self.sampleIndices[self.sampleIndiceIndex]
                        )
                    )
                    self.sampleIndiceIndex = (
                        None
                        if self.sampleIndices is None
                        else (
                            self.sampleIndiceIndex + 1
                            if self.sampleIndiceIndex + 1 < len(self.sampleIndices)
                            else 0  # re-iterate
                        )
                    )
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (
                self.max_trajectories is not None
                and len(completed_traj_infos) >= self.max_trajectories
            ):
                break
        return completed_traj_infos
