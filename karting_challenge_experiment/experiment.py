import logging
import numpy as np
import os
import random
import torch

from karting_challenge_rl.agent import Agent

logger = logging.getLogger("root")


class Experiment:
    """
    Represents an experiment with its main components, namely an environment
    for training and evaluating each and a reinforcement learning agent.
    An instance of this class (and all inheriting classes) read(s) the necessary
    fields of a given configuration (dict) and initializes the (both) environments
    and the agent. The corresponding configurable attributes are described
    in ./config/sample-config-*.yaml.
    """

    def __init__(self, config: dict):
        # access overall experiment parameters from configuration
        # access environment related parameters
        environment_config = config.get("environment")
        tracks_base_path = os.path.normpath(environment_config.get("tracks_base_path"))

        # setting the seed
        self.seed = environment_config.get("seed")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # initialize the environment
        self.env = [tracks_base_path]
        logger.info(f"Using the following {len(self.env)} tracks for learning: {self.env}")

        agent_config = config.get("agent")

        agent = Agent(
            save_path=config.get("experiment_path"),
            env=self.env,
            model_config=agent_config.get("model"),
            seed=self.seed,
            time_scale=environment_config.get("time_scale"),
            train_rollouts=agent_config.get("train_rollouts"),
            eval_rollouts=agent_config.get("eval_rollouts"),
            log_interval=agent_config.get("log_interval"),
            save_model_interval=agent_config.get("save_model_interval"),
            save_stats_interval=agent_config.get("save_stats_interval"),
            eval_interval=agent_config.get("eval_interval"),
            sb3_logger_config=agent_config.get("sb3_logger_config")
        )

        self.agent = agent

        logger.info("Successfully initialized training experiment")

    def conduct(self, mode="train"):
        if mode == "train":
            logger.info("Starting training experiment...")
            logger.info("Learning the agent...")
            self.agent.train()
            logger.info("Finished learning the agent")
        elif mode == "eval":
            logger.info("Starting evaluation experiment...")
            logger.info("Evaluating the agent...")
            self.agent.eval(close=True)
            logger.info("Finished evaluating the agent")
        else:
            raise ValueError(f"Unknown mode: {mode}")
