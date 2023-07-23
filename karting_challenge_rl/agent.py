import datetime
import json
import logging
import os
import re

import pandas as pd
import stable_baselines3
import torch
import torch.onnx

from itertools import chain
from karting_challenge_rl.models.ppo import PPO
from karting_challenge_rl.models.random_choice import Random
from matplotlib import pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from pathlib import Path
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import List, OrderedDict
from karting_challenge_rl.plotting import plot_line, calculate_values_from_data_for_plots

logger = logging.getLogger("root")


class ResultMemory:
    def __init__(self: "ResultMemory"):
        logger.info("Initializing result memory...")
        self.rewards_per_episode_per_rollout = {}
        self.timesteps_per_episode_per_rollout = {}
        self.checkpoints_passed_per_episode_per_rollout = {}
        self.action_dist_per_episode_per_rollout = {}

    def add(
            self,
            rollout_nr,
            rewards_per_episode_per_rollout,
            timesteps_per_episode_per_rollout,
            checkpoints_passed_per_episode_per_rollout,
            action_dist_per_episode_per_rollout
    ):
        self.rewards_per_episode_per_rollout[rollout_nr] = rewards_per_episode_per_rollout
        self.timesteps_per_episode_per_rollout[rollout_nr] = timesteps_per_episode_per_rollout
        self.checkpoints_passed_per_episode_per_rollout[rollout_nr] = checkpoints_passed_per_episode_per_rollout
        self.action_dist_per_episode_per_rollout[rollout_nr] = action_dist_per_episode_per_rollout

    def __len__(self):
        return len(self.rewards_per_episode_per_rollout)


class Agent:
    def __init__(
            self: "Agent",
            save_path: str,
            env: List[str],
            train_rollouts: int = 250,
            eval_rollouts: int = 10,
            log_interval: int = 10,
            save_model_interval: int = 10,
            save_stats_interval: int = 10,
            eval_interval: int = 50,
            model_config: dict = None,
            seed: int = 0,
            time_scale: float = 1.0,
            sb3_logger_config: List[str] = None,
    ):
        logger.info("Initializing agent...")

        # initialize overall agent attributes
        self.train_rollouts = train_rollouts
        self.eval_rollouts = eval_rollouts
        self.log_interval = log_interval
        self.save_model_interval = save_model_interval
        self.save_stats_interval = save_stats_interval
        self.eval_interval = eval_interval
        self.sb3_logger_config = sb3_logger_config

        # initialize paths
        self.model_save_path = os.path.join(save_path, "model")
        self.stats_save_path = os.path.join(save_path, "stats")
        self.best_save_path = os.path.join(self.model_save_path, "best")
        self.plots_save_path = os.path.join(save_path, "plots")
        self.seed = seed
        self.time_scale = time_scale
        self.model_config = model_config
        torch.manual_seed(self.seed)

        # initialize environment
        self.channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name=env[0], side_channels=[self.channel], seed=self.seed)
        self.channel.set_configuration_parameters(time_scale=self.time_scale)

        # initialize result memories to remember data
        self.train_result_memory = ResultMemory()
        self.eval_result_memory = ResultMemory()

        # initialize device if it is known
        self._init_device(model_config)

        # initialize model
        self._init_model(model_config)

        logger.info("Successfully initialized agent")

    def _init_device(self, model_config):
        """
        Initializes the PyTorch device to be used for training the agent.
        Checks if the device name given as hyperparameter device is known and allowed.
        If "auto" is provided, it checks whether CUDA is available or not.
        Currently, only available for sb3 models.
        """
        device_name = model_config.get("device")
        devices = ["cpu", "cuda", "auto"]

        if device_name not in devices:
            msg = f"Unknown device name: {device_name}"
            logger.error(msg)
            raise ValueError(msg)

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_name in ["auto", "cuda"]
            else torch.device(device_name)
        )

        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning(
                f"Specified device cuda but cuda is not available! "
                f"torch.cuda.is_available()=={torch.cuda.is_available()}"
            )
        logger.info(f"Using device {self.device}")

    def _init_model(self, model_config):
        """
        Initializes the model corresponding to the given model name.
        Checks if the model name and the policy name are known and allowed.
        Creates the model classes based on the given hyperparameters.
        """
        model_parameters = model_config.get("parameters")

        # initialize n_steps (equals buffer size) and corresponding total timesteps
        self.n_steps = model_parameters.get("n_steps")
        self.total_timesteps = self.train_rollouts * self.n_steps

        # initialize pretrained attributes if given
        self.pretrained_path = model_parameters.get("pretrained_path")

        # initialize model and policy identifiers
        self.rl_model_id = model_config.get("name")
        self.policy_id = model_parameters.get("policy")

        # supported policies and models
        policies = ["MlpPolicy"]
        models = {
            "PPO": self._init_ppo,
            "PPO-sb3": self._init_ppo_sb3,
            "random": self._init_random
        }

        # check compatibility of specified rl_model_id
        if self.rl_model_id not in models.keys():
            msg = f"Unsupported RL-Model: {self.rl_model_id}"
            logger.error(msg)
            raise ValueError(msg)

        # check compatibility of specified policy_id
        if self.policy_id not in policies:
            msg = f"Unknown policy id: {self.policy_id}"
            logger.error(msg)
            raise ValueError(msg)

        # initialize model
        models[self.rl_model_id](model_parameters)

        logger.info(f"Using model {self.rl_model_id} with policy {self.policy_id}.")

    def _init_ppo(self, model_parameters):
        self.pretrained_rollouts = int(Path(self.pretrained_path).stem) if self.pretrained_path else 0

        self.model = PPO(
            learning_rate=model_parameters.get("learning_rate"),
            gamma=model_parameters.get("gamma"),
            n_steps=self.n_steps,
            n_epochs=model_parameters.get("n_epochs"),
            obs_size=model_parameters.get("obs_size"),
            hidden_dim=model_parameters.get("hidden_dim"),
            additional_learnable_actions=model_parameters.get("additional_learnable_actions"),
            cov_mat_diag=model_parameters.get("cov_mat_diag"),
            cov_decay_mechanism=model_parameters.get("cov_decay_mechanism"),
            cov_decay_rate=model_parameters.get("cov_decay_rate"),
            cov_end_val=model_parameters.get("cov_end_val"),
            clip_range=model_parameters.get("clip_range"),
            policy_id=self.policy_id,
            model_path=self.pretrained_path
        )

    def _init_ppo_sb3(self, model_parameters):
        if self.pretrained_path:
            numbers = [int(s) for s in re.findall(r'\b\d+\b', self.pretrained_path)]
            self.pretrained_rollouts = numbers[-1]
        else:
            self.pretrained_rollouts = 0

        # wrap the unity environment with a gym environment
        self.gym_env = UnityToGymWrapper(self.env)

        if not self.pretrained_path:
            # initialize sb3 PPO model
            self.model = stable_baselines3.PPO(
                device=self.device,
                policy=self.policy_id,
                env=self.gym_env,
                seed=self.seed,
                verbose=1,
                learning_rate=model_parameters.get("learning_rate"),
                gamma=model_parameters.get("gamma"),
                n_steps=self.n_steps,
                batch_size=self.n_steps,  # batch as big as n_steps like in our implementation
                n_epochs=model_parameters.get("n_epochs"),
                clip_range=model_parameters.get("clip_range")
            )
        else:
            self.model = stable_baselines3.PPO.load(path=self.pretrained_path, device=self.device, env=self.gym_env)

        # initialize logger (logging also means persisting in sb3)
        sb3_logger = (configure(self.stats_save_path, self.sb3_logger_config) if self.sb3_logger_config else None)
        self.model.set_logger(sb3_logger)

        # initialize callback for model saving
        self.save_model_callback = CheckpointCallback(
            save_freq=self.save_model_interval * self.n_steps,
            save_path=self.model_save_path,
        )

    def _init_random(self, model_parameters):
        self.model = Random(
            n_steps=self.n_steps,
            additional_learnable_actions=model_parameters.get("additional_learnable_actions")
        )

    def train(self):
        trainers = {
            "PPO": self._train_ppo,
            "PPO-sb3": self._train_ppo_sb3,
            "random": self._train_random
        }
        trainers[self.rl_model_id]()

    def _train_ppo(self):
        # Let the agent learn with an own model implementation.
        # A rollout represents a trajectory of n_steps of the agent in the environment.
        # Such a rollout consists of 1 or more episodes.
        # An episode represents a trajectory until the agent somehow fails in the environment
        # or the given timestep limit (via hyperparameter episode_timesteps) is reached.
        start = 1 + self.pretrained_rollouts
        stop = self.train_rollouts + self.pretrained_rollouts
        rollout_range = range(start, stop + 1)
        for rollout in rollout_range:
            # train the agent's model
            logger.info(
                f"Starting train rollout {rollout} (i.e. total timesteps from "
                f"{(rollout - 1) * self.n_steps} to {rollout * self.n_steps})...")

            rollout_result = self.model.train(rollout - self.pretrained_rollouts, self.env)

            # remember training results from this rollout
            self.train_result_memory.add(rollout, *rollout_result)

            if rollout % self.log_interval == 0 or rollout == stop:
                self._log_rollout_summary(rollout, rollout_result)

            if rollout % self.save_model_interval == 0 or rollout == stop:
                self._save_model(rollout_nr=rollout)

            if rollout % self.save_stats_interval == 0 or rollout == stop:
                self._save_stats(rollout_nr=rollout)

            if rollout % self.eval_interval == 0 or rollout == stop:
                self.eval(rollout_nr=rollout)

        # close all tracks in env
        self.env.close()
        self._plot_training_infos()
        self._plot_action_dist()

    def _train_ppo_sb3(self):
        self.model.learn(
            total_timesteps=self.total_timesteps,
            log_interval=self.log_interval,
            callback=self.save_model_callback
        )
        self.model.save(os.path.join(self.model_save_path, "final"))
        self._plot_sb3_rewards()
        self.gym_env.close()

    def _train_random(self):
        start = 1
        stop = self.train_rollouts
        rollout_range = range(start, stop + 1)

        for rollout in rollout_range:
            rollout_result = self.model.run(rollout, self.env)
            self.train_result_memory.add(rollout, *rollout_result)
            self._save_stats(rollout_nr=rollout)

        self.env.close()
        self._plot_training_infos(random=True)
        self._plot_action_dist()

    @staticmethod
    def _log_rollout_summary(rollout_nr, rollout_result):
        logger.info(f"===== Summary of rollout {rollout_nr} =====")
        logger.info(
            f"cumulated reward = {sum(list(chain.from_iterable(list(rollout_result[0].values()))))}")
        logger.info(f"number of episodes = {len(rollout_result[1])}")
        logger.info(
            f"mean checkpoints passed = "
            f"{sum(list(rollout_result[2].values())) / len(rollout_result[2].values())}")
        logger.info(f"===== End of summary =====")

    def eval(self, rollout_nr: int = 0, close: bool = False):
        """
        Evaluation uses tests on an environment and collects data that is plotted.
        """
        logger.info(f"===== Starting Evaluation Phase =====")

        evaluators = {
            "PPO": self._eval_ppo,
            "PPO-sb3": self._eval_ppo_sb3,
            "random": self._eval_random
        }
        evaluators[self.rl_model_id](rollout_nr)

        self.env.close() if close else None
        logger.info(f"===== End of Evaluation Phase =====")

    def _eval_ppo(self, rollout_nr):
        start = 1
        stop = self.eval_rollouts
        rollout_range = range(start, stop + 1)
        for rollout in rollout_range:
            logger.info(f"Starting eval rollout {rollout} (i.e. total timesteps from "
                        f"{(rollout - 1) * self.n_steps} to {rollout * self.n_steps})...")

            rollout_result = self.model.train(rollout, self.env, eval=True)

            # remember eval results of this rollout
            self.eval_result_memory.add(rollout_nr + rollout, *rollout_result)

            self._save_stats(rollout_nr, eval=True)

        self._plot_training_infos(eval=True, rollout_nr=rollout_nr)

    def _eval_ppo_sb3(self, rollout_nr):
        obs = self.gym_env.reset()

        for rollout in range(0, self.eval_rollouts):
            eval_rewards = []
            for step in range(0, self.n_steps):
                action, _states = self.model.predict(obs)
                obs, rewards, done, info = self.gym_env.step(action)
                eval_rewards.append(rewards)
                # self.gym_env.render("human")
                obs = obs if not done else self.gym_env.reset()

            # remember eval results of this rollout
            self.eval_result_memory.add(rollout_nr + rollout, eval_rewards, None, None, None)

        logger.warning("There is no plotting functionality for the evaluation results of PPO-sb3 implemented yet.")

    @staticmethod
    def _eval_random():
        logger.warning("Evaluating random is not further defined")

    def _save_model(self, rollout_nr: int):
        """
        Saves a .pt and an .onnx model each save_model_interval and at the end of all rollouts.
        Save model as onnx: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html.
        """
        save_path_with_rollout_nr = os.path.join(self.model_save_path, str(rollout_nr))
        logger.info(f"Saving model of rollout {rollout_nr}")
        # save as pt-file
        torch.save(self.model.policy.state_dict(), save_path_with_rollout_nr + ".pt")
        # export model as onnx-file 
        brain = KartingBrain(self.model.policy.actor, 3)
        torch.onnx.export(brain,
                          torch.randn(1, 48),
                          save_path_with_rollout_nr + "KartingBrain.onnx",
                          export_params=True,
                          opset_version=11,
                          input_names=["obs_0"],
                          verbose=True,
                          output_names=['continuous_actions', 'continuous_action_output_shape',
                                        'version_number', 'memory_size'],
                          dynamic_axes={'obs_0': {0: 'batch'},  # 'vector_observation': {0: 'batch'},
                                        'continuous_actions': {0: 'batch'},
                                        'continuous_action_output_shape': {0: 'batch'}}
                          )

    def _save_stats(self, rollout_nr: int, eval: bool = False):
        """
        Saves the stats in a json file with respective name.
        """
        logger.info(f"Saving stats of rollout {rollout_nr}")

        result_memory = self.eval_result_memory if eval else self.train_result_memory
        filename_to_data_map = {
            "rewards_per_rollout.json": result_memory.rewards_per_episode_per_rollout,
            "timesteps_per_episode_per_rollout.json": result_memory.timesteps_per_episode_per_rollout,
            "checkpoints_passed_per_episode_per_rollout.json": result_memory.checkpoints_passed_per_episode_per_rollout,
            "action_distribution.json": result_memory.action_dist_per_episode_per_rollout
        }

        prefix = "eval_" if eval else ""
        for filename, data in filename_to_data_map.items():
            path = os.path.join(self.stats_save_path, prefix + filename)
            with open(path, 'w') as file:
                file.write(json.dumps(str(data)))

    def _plot_training_infos(self, eval: bool = False, rollout_nr: int = 0, random: bool = False):
        # prepare the reward-related data that will be plotted
        plot_data = OrderedDict()

        if eval:
            rewards_per_episode_per_rollout = self.eval_result_memory.rewards_per_episode_per_rollout
            timesteps_per_episode_per_rollout = self.eval_result_memory.timesteps_per_episode_per_rollout
            checkpoints_passed_per_episode_per_rollout = self.eval_result_memory.checkpoints_passed_per_episode_per_rollout
        else:
            rewards_per_episode_per_rollout = self.train_result_memory.rewards_per_episode_per_rollout
            timesteps_per_episode_per_rollout = self.train_result_memory.timesteps_per_episode_per_rollout
            checkpoints_passed_per_episode_per_rollout = self.train_result_memory.checkpoints_passed_per_episode_per_rollout

        output = calculate_values_from_data_for_plots(rewards_per_episode_per_rollout.values(),
                                                      timesteps_per_episode_per_rollout.values(),
                                                      checkpoints_passed_per_episode_per_rollout.values())

        plot_data['cum_rewards'] = output[0]
        plot_data['mean_episode_rewards'] = output[1]
        plot_data['smoothed_rewards'] = output[2]
        plot_data['mean_steps_per_episode'] = output[3]
        plot_data['episodes'] = output[4]
        plot_data['mean_ckeckpoints_per_episode'] = output[5]

        now = datetime.datetime.now().strftime("%m/%d/%Y")
        fontsize = 20

        fig, axs = plt.subplots(6, 1)
        sup_title = f"Training {now}" if not random else f"Random Agent {now}"
        fig.suptitle(sup_title, fontsize=fontsize)
        fig.set_figwidth(15)
        fig.set_figheight(25)

        y_labels = ['Cumulated Episode Rewards', 'Mean Episode Rewards', 'Smoothed Rewards', 'Mean Steps / Episode',
                    'Number of Episodes', 'Mean Checkpoints / Episode']
        plot_line(axs, plot_data, "", y_labels, self.train_rollouts, self.n_steps, fontsize)
        plt.tight_layout()
        if not eval:
            fig.savefig(os.path.join(self.plots_save_path, "all_in_one.png"))
        else:
            fig.savefig(os.path.join(self.plots_save_path, str(rollout_nr) + "all_in_one_eval.png"))

    def _plot_action_dist(self):
        actions = self.train_result_memory.action_dist_per_episode_per_rollout

        model_params = self.model_config.get("parameters")
        learnable_actions = model_params.get("additional_learnable_actions")

        now = datetime.datetime.now().strftime("%m/%d/%Y")
        fontsize = 20

        if "acceleration" in learnable_actions and "breaking" in learnable_actions:
            nr_actions_learned = 3
        elif "acceleration" not in learnable_actions and "breaking" not in learnable_actions:
            nr_actions_learned = 1
        else:
            nr_actions_learned = 2

        fig, axs = plt.subplots(nr_actions_learned, 1)
        fig.suptitle('Action distribution' + now, fontsize=fontsize)
        fig.set_figheight(17)

        act_list = []

        for episode in actions.values():
            for item in episode.values():
                act_list.append(item)

        if nr_actions_learned > 1:
            axs[0].plot(list(item[0] for item in act_list))
            axs[0].set_xlabel("Episodes", fontsize=fontsize)
            axs[0].set_ylabel('Mean steering', fontsize=fontsize)
            axs[0].grid(True)

            if "acceleration" in learnable_actions:
                axs[1].plot(list(item[1] for item in act_list))
                axs[1].set_xlabel("Episodes", fontsize=fontsize)
                axs[1].set_ylabel('Mean acceleration', fontsize=fontsize)
                axs[1].grid(True)
                if "breaking" in learnable_actions:
                    axs[2].plot(list(item[2] for item in act_list))
                    axs[2].set_xlabel("Episodes", fontsize=fontsize)
                    axs[2].set_ylabel('Mean breaking', fontsize=fontsize)
                    axs[2].grid(True)

            elif "breaking" in learnable_actions:
                axs[1].plot(list(item[1] for item in act_list))
                axs[1].set_xlabel("Episodes", fontsize=fontsize)
                axs[1].set_ylabel('Mean breaking', fontsize=fontsize)
                axs[1].grid(True)

        else:
            axs.plot(list(item[0] for item in act_list))
            axs.set_xlabel("Episodes", fontsize=fontsize)
            axs.set_ylabel('Mean steering', fontsize=fontsize)
            axs.grid(True)

        fig.savefig(os.path.join(self.plots_save_path, "action_dist.png"))

    def _plot_sb3_rewards(self):
        # available: ['time/total_timesteps',
        #             'time/fps',
        #             'time/iterations',
        #             'rollout/ep_len_mean',
        #             'time/time_elapsed',
        #             'rollout/ep_rew_mean',
        #             'train/clip_range',
        #             'train/entropy_loss',
        #             'train/learning_rate',
        #             'train/n_updates',
        #             'train/policy_gradient_loss',
        #             'train/loss',
        #             'train/explained_variance',
        #             'train/approx_kl',
        #             'train/std',
        #             'train/clip_fraction',
        #             'train/value_loss']
        df = pd.read_csv(self.stats_save_path + '\\\\progress.csv',
                         usecols=['time/total_timesteps', 'rollout/ep_rew_mean'])
        timesteps = df['time/total_timesteps'].tolist()
        episode_reward_mean = df['rollout/ep_rew_mean'].tolist()
        now = datetime.datetime.now().strftime("%m/%d/%Y")
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f'SB3 rewards of run {now}')
        plt.plot(timesteps, episode_reward_mean)
        fig.savefig(os.path.join(self.plots_save_path, "ep_rew_mean.png"))


class KartingBrain(torch.nn.Module):
    def __init__(self, policy, output_size: 3):
        super().__init__()
        self.policy = policy
        version_number = torch.Tensor([3])  # version_number
        self.version_number = torch.nn.Parameter(version_number, requires_grad=False)
        memory_size = torch.Tensor([0])  # memory_size
        self.memory_size = torch.nn.Parameter(memory_size, requires_grad=False)
        output_shape = torch.Tensor([output_size])  # continuous_action_output_shape
        self.output_shape = torch.nn.Parameter(output_shape, requires_grad=False)

    def forward(self, observation):
        action_pred = self.policy(observation)
        return action_pred, self.output_shape, self.version_number, self.memory_size
