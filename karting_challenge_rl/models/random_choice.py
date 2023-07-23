import logging
import random
import numpy as np
import torch

from karting_challenge_rl.models.misc import prepare_action_stats
from mlagents_envs.base_env import ActionTuple

logger = logging.getLogger("root")


class Random:
    def __init__(
            self,
            n_steps=2048,
            additional_learnable_actions=[]
    ):
        logger.info("Initializing random model...")

        # initialize overall parameters 
        self.n_steps = n_steps

        # initialize learnable actions
        self.learnable_actions = ["steering"] + additional_learnable_actions
        self.actions_dim = len(self.learnable_actions)
        actions = ["steering", "acceleration", "breaking"]
        for action in self.learnable_actions:
            if action not in actions:
                msg = f"Unknown action defined: {action}"
                logger.error(msg)
                raise ValueError(msg)
        logger.info(f"Using learnable actions {self.learnable_actions}, actions_dim = {self.actions_dim}")

    def run(self, rollout_nr, track):
        logger.info(f"Starting rollout {rollout_nr} (i.e. total timesteps = {rollout_nr * self.n_steps})...")

        behavior_name = None

        # keep track of the following per rollout
        episode_nr = -1  # starts with -1 such that the first episode is 0 (and so this number equals the # crashes)
        rollout_timestep = 0  # rollout-wise timestep counter (for logging purposes)
        current_episode_timestep = 0  # episode-wise timestep counter
        checkpoints_passed = 0  # episode-wise checkpoint counter
        action_list = []
        episode_rewards = []

        rollout_rewards_per_episode = {}
        rollout_timesteps_per_episode = {}
        rollout_checkpoints_passed_per_episode = {}
        rollout_action_dist_per_episode = {}

        # Run the episodes of this rollout (multiple episodes per rollout if the agent crahes).
        # One rollout lasts # n_steps timesteps (thus, one episode lasts <= n_steps timesteps).
        rollout_done = False
        track_done = True
        while not rollout_done:
            # initialize the track (placement of agent on track)
            if track_done:
                # starting next episode in this rollout, i.e. increment episode number
                episode_nr += 1
                # reset the epsiode-wise timestep couter
                current_episode_timestep = 0
                # reset the checkpoints passed measure
                checkpoints_passed = 0
                # reset the array for the episode rewards
                episode_rewards = []
                # reset the environment
                track.reset()

                # get the agents behavior group
                behavior_name = list(track.behavior_specs)[0]
                spec = track.behavior_specs[behavior_name]
                logger.debug(f"Using track: {spec}")

                # get list of all active agents
                decision_steps, terminal_steps = track.get_steps(behavior_name)

                action_list = []

            # increment timesteps
            rollout_timestep += 1
            current_episode_timestep += 1

            action, action_distribution_list = self._prepare_action()
            action_list.append(action_distribution_list)

            track.set_actions(behavior_name, action)
            track.step()

            # get list of all active agents 
            decision_steps, terminal_steps = track.get_steps(behavior_name)

            # done if if no agent is available for an action or there is an agent that has terminated after the step
            track_done = len(decision_steps) <= 0 or len(terminal_steps) > 0

            # get the reward of the action in the last step
            # this reward comes from terminal_steps if the agent did the last timestep of this rollout
            rew = decision_steps[0].reward if not track_done else terminal_steps[0].reward
            episode_rewards.append(rew)

            checkpoints_passed += 1 if rew >= 1 else checkpoints_passed

            # done if done (see above) or the maximum number of timesteps is reached
            rollout_done = rollout_timestep == self.n_steps

            if track_done or rollout_done:
                rollout_rewards_per_episode[episode_nr] = episode_rewards
                rollout_timesteps_per_episode[episode_nr] = current_episode_timestep
                rollout_checkpoints_passed_per_episode[episode_nr] = checkpoints_passed
                action_list = prepare_action_stats(action_list)
                rollout_action_dist_per_episode[episode_nr] = action_list

        return tuple((rollout_rewards_per_episode,
                      rollout_timesteps_per_episode,
                      rollout_checkpoints_passed_per_episode,
                      rollout_action_dist_per_episode))

    def _prepare_action(self):

        # always learn to steer
        steering = random.uniform(-1, 1)

        # learn to accelerate and/or to break depending on hyperparameters
        if "acceleration" in self.learnable_actions and "breaking" in self.learnable_actions:
            acceleration = random.uniform(-1, 1)
            breaking = random.uniform(-1, 1)
            action_distribution_list = [steering, acceleration, breaking]
        elif "acceleration" in self.learnable_actions:
            acceleration = random.uniform(-1, 1)
            breaking = 0
            action_distribution_list = [steering, acceleration]
        elif "breaking" in self.learnable_actions:
            acceleration = 1
            breaking = random.uniform(-1, 1)
            action_distribution_list = [steering, breaking]
        else:
            acceleration = 1
            breaking = 0
            action_distribution_list = [steering]

        action = [steering, acceleration, breaking]

        action = torch.tensor([action])
        action = np.array(action, dtype=np.dtype('f'))
        action = ActionTuple(action)

        return action, action_distribution_list
