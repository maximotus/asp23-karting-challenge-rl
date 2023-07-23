import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from karting_challenge_rl.models.misc import calculate_advantages, calculate_returns, init_weights, prepare_action_stats
from mlagents_envs.base_env import ActionTuple
from torch.distributions import MultivariateNormal

logger = logging.getLogger("root")


class N(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.net(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, obs):
        action_pred = self.actor(obs)
        value_pred = self.critic(obs)
        return action_pred, value_pred


class ReplayBuffer:
    def __init__(self):
        self.memory = []
        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.new_obs = None,

    def add(self, obs, values, actions, rewards, log_prob_action, done, new_obs):
        self.states.append(obs)
        self.values.append(values)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.log_prob_actions.append(log_prob_action)
        self.dones.append(done)
        self.new_obs = new_obs

    def sample(self):
        return self.states, self.actions, self.rewards, self.values, self.log_prob_actions, self.dones, self.new_obs

    def buffer_length(self):
        return len(self.states)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.values = []
        self.rewards = []


class PPO():
    def __init__(
            self,
            learning_rate=0.0003,
            gamma=0.98,
            n_steps=2048,
            n_epochs=5,
            obs_size=48,
            hidden_dim=128,
            additional_learnable_actions=[],
            cov_mat_diag=0.5,
            cov_decay_mechanism="exp",
            cov_decay_rate=0.01,
            cov_end_val=1e-6,
            clip_range=0.2,
            model_path=None,
            policy_id="MlpPolicy"
    ):
        logger.info("Initializing PPO model...")

        # initialize overall parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.cov_decay_mechanism = cov_decay_mechanism
        self.cov_decay_rate = cov_decay_rate
        self.cov_end_val = cov_end_val
        self.clip_range = clip_range
        self.criterion = nn.MSELoss()
        self.model_path = model_path

        # initialize rollout buffer
        self.buffer = ReplayBuffer()

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

        # initialize covariance matrix for querying the actor
        self.cov_mat_diag = cov_mat_diag
        cov_var = torch.full(size=(self.actions_dim,), fill_value=self.cov_mat_diag)
        self.cov_mat = torch.diag(cov_var)
        logger.info(f"Initialized covariance matrix used to query the actor for actions: {self.cov_mat}")

        # initialize policy
        if policy_id == "MlpPolicy":
            INPUT_DIM = obs_size  # state space
            HIDDEN_DIM = hidden_dim  # batch space
            ACTOR_OUTPUT_DIM = self.actions_dim  # action space
            CRITIC_OUTPUT_DIM = 1
            WEIGHT_DECAY = 1e-6

            # initialize two models with same input but different output dimensions
            actor = N(INPUT_DIM, HIDDEN_DIM, ACTOR_OUTPUT_DIM)
            critic = N(INPUT_DIM, HIDDEN_DIM, CRITIC_OUTPUT_DIM)
            self.policy = ActorCritic(actor, critic)
            self.policy.apply(init_weights) if not self.model_path else self._load_policy()
            self.optimizer_actor = optim.Adam(self.policy.actor.parameters(), lr=self.learning_rate,
                                              weight_decay=WEIGHT_DECAY)
            self.optimizer_critic = optim.Adam(self.policy.critic.parameters(), lr=self.learning_rate,
                                               weight_decay=WEIGHT_DECAY)

        logger.info(
            f"Initialized PPO model with {policy_id} and {sum(1 for _ in self.policy.parameters())} optimization parameters.")
        logger.info("Optimizable parameters:")
        for param in self.policy.parameters():
            logger.info("    " + str(type(param)) + str(param.size()))

    def train(self, rollout_nr, track, eval: bool = False):
        # initialize obs and behavior name, will be overwritten on episode starts
        obs = None
        behavior_name = None

        # keep track of the following per rollout
        episode_nr = 0  # starts with 0 such that the first episode is 1 (and so this number equals the # crashes)
        rollout_timestep = 0  # rollout-wise timestep counter (for logging purposes)
        current_episode_timestep = 0  # episode-wise timestep counter
        checkpoints_passed = 0  # episode-wise checkpoint counter
        episode_rewards = []
        episode_action_dist = []
        rollout_rewards_per_episode = {}
        rollout_timesteps_per_episode = {}
        rollout_checkpoints_passed_per_episode = {}
        rollout_action_dist_per_episode = {}

        # calculate diagonal values of the covariance matrix used for sampling from a normal distribution
        val = self._calculate_decayed_val(rollout_nr - 1, eval)
        logger.info(f"The diagonal values used for sampling are val = {val}")

        # Run the episodes of this rollout (multiple episodes per rollout if the agent crashes).
        # One rollout lasts # n_steps timesteps (thus, one episode lasts <= n_steps timesteps).
        rollout_done = False
        track_done = True
        while not rollout_done:
            # initialize the track (placement of agent on track)
            if track_done:
                # starting next episode in this rollout, i.e. increment episode number
                episode_nr += 1

                # reset the episode-wise timestep counter
                current_episode_timestep = 0

                # reset the checkpoints passed measure
                checkpoints_passed = 0

                # reset the array for the episode rewards
                episode_rewards = []

                # reset the environment
                track.reset()

                # get the agents' behavior group
                behavior_name = list(track.behavior_specs)[0]
                spec = track.behavior_specs[behavior_name]
                logger.debug(f"Using track: {spec}")

                # enable train mode of policy model
                self.policy.train()

                # get list of all active agents
                decision_steps, terminal_steps = track.get_steps(behavior_name)

                # get the initial observation
                obs = decision_steps.obs
                logger.debug(f"raw observation = {obs}")
                obs = torch.FloatTensor(np.array(obs))
                logger.debug(f"observation = {obs}")

                episode_action_dist = []

            # increment timesteps
            rollout_timestep += 1
            current_episode_timestep += 1

            # get the predictions for this observation from the ActorCritic
            action_pred, value_pred = self.policy(obs)
            cov_var = torch.full(size=(self.actions_dim,), fill_value=val)
            self.cov_mat = torch.diag(cov_var)
            action_dist = MultivariateNormal(action_pred, self.cov_mat)
            action_sample = action_dist.sample()
            log_prob_action = action_dist.log_prob(action_sample)

            episode_action_dist.append(action_sample.tolist()[0])
            # execute the action in the environment
            action = self._prepare_action(action_sample)
            track.set_actions(behavior_name, action)
            track.step()

            # get list of all active agents 
            decision_steps, terminal_steps = track.get_steps(behavior_name)

            # done if if no agent is available for an action or there is an agent that has terminated after the step
            track_done = len(decision_steps) <= 0 or len(terminal_steps) > 0

            # get the new observation after the step in the environment
            # this observation comes from terminal_steps if the agent is done in this rollout
            new_obs = decision_steps.obs if not track_done else terminal_steps.obs
            new_obs = torch.FloatTensor(np.array(new_obs))

            # get the reward of the action in the last step
            # this reward comes from terminal_steps if the agent did the last timestep of this rollout
            rew = decision_steps[0].reward if not track_done else terminal_steps[0].reward
            self.buffer.add(obs, value_pred, action_sample, rew, log_prob_action, track_done, new_obs)
            episode_rewards.append(rew)

            if rew >= 1:
                checkpoints_passed += 1

            # done if done (see above) or the maximum number of timesteps is reached
            rollout_done = self.buffer.buffer_length() == self.n_steps

            if track_done or rollout_done:
                rollout_rewards_per_episode[episode_nr] = episode_rewards
                rollout_timesteps_per_episode[episode_nr] = current_episode_timestep
                rollout_checkpoints_passed_per_episode[episode_nr] = checkpoints_passed
                action_list = prepare_action_stats(episode_action_dist)
                rollout_action_dist_per_episode[episode_nr] = action_list

            # logging for debugging purposes
            logger.debug(f"step {rollout_timestep}")
            logger.debug(f"action_dist = {action_dist}")
            logger.debug(f"action_pred = {action_pred}")
            logger.debug(f"action = {action}")
            logger.debug(f"value_pred = {value_pred}")
            logger.debug(f"action_sample = {action_sample}")
            logger.debug(f"log_prob_action = {log_prob_action}")
            logger.debug(f"old observation = {obs}")
            logger.debug(f"new observation = {new_obs}")

            # overwrite old observation
            obs = new_obs

        # update the policy model if the rollout is done
        states, actions, rewards, values, log_prob_actions, dones, new_state = self.buffer.sample()
        self.buffer.clear()

        if not eval:
            self._update_policy(states, actions, rewards, values, log_prob_actions, dones, new_state)

        return tuple((rollout_rewards_per_episode,
                      rollout_timesteps_per_episode,
                      rollout_checkpoints_passed_per_episode,
                      rollout_action_dist_per_episode))

    def _calculate_decayed_val(self, rollout_nr, eval):
        """
        Calculates the decayed diagonal value for the covariance matrix
        based on the current rollout_nr and the defined decay mechanism.
        """
        if self.cov_decay_mechanism == "lin":
            val = self.cov_mat_diag - self.cov_decay_rate * rollout_nr
        elif self.cov_decay_mechanism == "exp":
            val = self.cov_mat_diag * math.exp(- self.cov_decay_rate * rollout_nr)
        elif self.cov_decay_mechanism == "static":
            val = self.cov_mat_diag
        else:
            raise ValueError("Unknown value for hyper-parameter cov_decay_mechanism given.")
        return val if val > 0.0 and not eval else self.cov_end_val

    def _prepare_action(self, action_sample):
        """
        Prepares the action for the environment.
        """
        # always learn to steer
        steering = action_sample[0, 0]

        # learn to accelerate and/or to break depending on hyperparameters
        if "acceleration" in self.learnable_actions and "breaking" in self.learnable_actions:
            acceleration = action_sample[0, 1]
            breaking = action_sample[0, 2]
        elif "acceleration" in self.learnable_actions:
            acceleration = action_sample[0, 1]
            breaking = 0
        elif "breaking" in self.learnable_actions:
            acceleration = 1
            breaking = action_sample[0, 1]
        else:
            acceleration = 1
            breaking = 0

        action = torch.tensor([[steering, acceleration, breaking]])
        action = np.array(action, dtype=np.dtype('f'))
        return ActionTuple(action)

    def _update_policy(self, states, actions, rewards, values, log_prob_actions, dones, new_state):
        """
        Updates the actor and critic network weights based on the
        clipped surrogate objective of PPO, see https://arxiv.org/pdf/1707.06347.pdf.
        """
        _, new_value = self.policy(new_state)

        # prepare the rollout sample
        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        values = torch.cat(values, 0)
        values = torch.cat((values, new_value), 0)

        log_prob_actions = torch.cat(log_prob_actions, 0)
        dones = ~np.array(dones)

        # calculate returns (i.e. discounted future rewards)
        returns = calculate_returns(rewards, values, self.gamma, dones, False)

        # Calculate advantages, i.e. difference between returns
        # (factual discounted future rewards)
        # and values (estimated discounted future rewards).
        advantages = calculate_advantages(returns, values, False)
        logger.debug(f"returns = {returns}")

        total_actor_loss = 0
        total_critic_loss = 0

        states = states.detach()
        actions = actions.detach()
        advantages = advantages.detach()
        returns = returns.detach()
        log_prob_actions = log_prob_actions.detach()

        for i in range(self.n_epochs):
            # get network prediction
            # action_pred := estimate of the best action
            # value_pred := estimate of discounted future rewards 
            action_pred, value_pred = self.policy(states)

            # get actions and ratios
            action_dist = MultivariateNormal(action_pred, self.cov_mat)
            new_log_prob_actions = action_dist.log_prob(actions)

            # Clipped surrogate objective (compare https://arxiv.org/pdf/1707.06347.pdf) on seperate networks.
            ratios = self._compute_log_prob_action_ratios(new_log_prob_actions, log_prob_actions)

            actor_loss = self._compute_actor_loss(ratios, advantages)
            self._optimize(self.optimizer_actor, actor_loss)

            critic_loss = self._compute_critic_loss(value_pred, returns)
            self._optimize(self.optimizer_critic, critic_loss)

            total_actor_loss += actor_loss.detach().numpy()
            total_critic_loss += critic_loss.detach().numpy()

            logger.debug(f"value_pred = {value_pred}")
            logger.debug(f"returns = {returns}")
            logger.debug(f"ratios = {ratios}")
            logger.debug(f"old_log_prob_actions = {log_prob_actions}")
            logger.debug(f"new_log_prob_actions = {new_log_prob_actions}")
            logger.debug(f"actor loss (clipped surrogate objective) = {actor_loss}")
            logger.debug(f"critic loss (clipped surrogate objective) = {actor_loss}")
            logger.debug(f"Finished gradient update step {i + 1}") if not eval else None

        logger.info(f"Updated the gradients for {self.n_epochs} epochs") if not eval else None

    @staticmethod
    def _compute_log_prob_action_ratios(new_log_prob_actions, log_prob_actions):
        """
        Calculates the ratios between the updated logarithmic action probabilities and the
        initial logarithmic action probabilities from the start of the update iterations.
        """
        return (new_log_prob_actions - log_prob_actions).exp()

    def _compute_actor_loss(self, ratios, advantages):
        """
        Computes the policy gradient loss (ratios is always positive).       
        1) Compute the standard policy gradient objective.
        2) Compute the clipped version of the standard policy gradient objective.
        "With this scheme, we only ignore the change in probability ratio when it would make the objective improve,
        and we include it when it makes the objective worse." (compare https://arxiv.org/pdf/1707.06347.pdf)
        """
        surrogate_loss_1 = ratios * advantages
        surrogate_loss_2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
        return - torch.min(surrogate_loss_1, surrogate_loss_2).mean()

    def _compute_critic_loss(self, value_pred, returns):
        """
        Compute critic loss (squared error loss, see https://arxiv.org/pdf/1707.06347.pdf).
        MSELoss already reduces to mean per default, see https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html.
        """
        return self.criterion(torch.flatten(value_pred).float(), returns.float())

    @staticmethod
    def _optimize(optimizer, loss):
        """
        Back-propagates the gradients of the given optimizer and loss function. 
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _load_policy(self):
        logger.info("Loading PPO policy model...")
        self.policy.load_state_dict(torch.load(self.model_path))
