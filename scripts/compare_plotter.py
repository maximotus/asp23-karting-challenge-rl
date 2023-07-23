import argparse
import os
import sys
import json
import yaml
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

from typing import OrderedDict
from itertools import zip_longest

# This part of the script adapts the Python sys.path so the karting_challenge_experiment package
# can be used like a package without packaging it.
# This is meant for development use-cases only.
# When development and debugging is done, the packages can be build with and installed via pip
# or conda and this part can be removed.

# Get the absolute path to the 'project' directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'project' directory to the Python path
sys.path.append(project_dir)

from karting_challenge_rl.plotting import plot_line, calculate_values_from_data_for_plots


def main():
    """
    IMPORTANT:
        mean-valued-data-csv's have to have "mean" in their name, the rest is not allowed to have this keyword
        mean-valued-data of stable-baseline3 has to have "sb" and "mean" in the name
        stable-baseline3-runs have to have sb3 in their config name to be recognized as such

    Fill values of  
        conf_paths:     paths to yml or csv's be plotted
        mean_name:      how you want to name your mean data
        create_mean:    set to True if you want to create mean data with "mean_data" as it's name
        fig_name:       hwo you want to name to plot

    Notice:
    if a path from stable-baselines3 (sb3) is included only the mean_reward will be plotted (the rest is not available)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create_mean",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="relative or absolute path to the configuration file inside the experiment directory",
    )
    parser.set_defaults(create_mean=False)
    parser.add_argument(
        "--conf_paths",
        type=str,
        nargs=argparse.ONE_OR_MORE,
        metavar="[PATH_TO_CONF_FILE_IN_EXP_DIR, ...]",
        required=True,
        help="relative or absolute path to the configuration file inside the experiment directory",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        metavar="RESULT_FILE_NAME",
        required=True,
        help="filename that saves either the mean data or the plot image",
    )
    args = parser.parse_args()

    create_mean = args.create_mean
    mean_name = "mean_" + args.file_name
    fig_name = args.file_name
    conf_paths = args.conf_paths

    print("Creating mean") if create_mean else print("Creating comparison plot")
    print(f"Saving as {mean_name}")
    for i, conf_path in enumerate(conf_paths):
        print(f"{i + 1}-th conf_path: {conf_path}")

    experiments, sb3 = _load_data(conf_paths)

    if create_mean:
        compare_data = {}
        for experiment in experiments:
            if "sb3" in experiment or "exp-2_" in experiment or "exp-4_" in experiment or "exp-6_" in experiment:
                compare_data.update(
                    {experiment: {'mean_episode_rewards': experiments[experiment]['episode_reward_mean']}})
            else:
                cum_rewards, mean_episode_rewards, smoothed_rewards, mean_steps_per_episode, episodes, mean_checkpoints_per_episode = calculate_values_from_data_for_plots(
                    experiments[experiment]["rewards_per_rollout"].values(),
                    experiments[experiment]["timesteps_per_episode_per_rollout"].values(),
                    experiments[experiment]["checkpoints_passed_per_episode_per_rollout"].values())

                compare_data.update(
                    {experiment: {'cum_rewards': cum_rewards, 'mean_episode_rewards': mean_episode_rewards,
                                  'smoothed_rewards': smoothed_rewards,
                                  'mean_steps_per_episode': mean_steps_per_episode,
                                  'episodes': episodes, 'mean_checkpoints_per_episode': mean_checkpoints_per_episode}})

        _calculate_mean_from_data(sb3, mean_name, compare_data)

    else:
        fig, axs = plt.subplots(6, 1) if not sb3 else plt.subplots(1, 1)
        now = datetime.datetime.now().strftime("%m/%d/%Y")
        fontsize = 20
        fig.suptitle('Training ' + now, fontsize=fontsize)
        fig.set_figwidth(15)
        fig.set_figheight(25) if not sb3 else None

        for experiment in experiments:
            label = experiment
            rollouts = experiments[experiment]['rollouts']
            n_steps = experiments[experiment]['n_steps']
            plot_data = OrderedDict()

            if "exp-2_" in label or "exp-4_" in label or "exp-6_" in label or "sb3" in label:
                plot_data['mean_episode_rewards'] = experiments[experiment]['episode_reward_mean']

            elif "mean" in label:
                if "sb" not in label:
                    df = experiments[experiment]['data']
                    plot_data['cum_rewards'] = df['mean_cum_rewards'].tolist()
                    plot_data['mean_episode_rewards'] = df['mean_mean_episode_rewards'].tolist()
                    plot_data['smoothed_rewards'] = df['mean_smoothed_rewards'].tolist()
                    plot_data['mean_steps_per_episode'] = df['mean_mean_steps_per_episode'].tolist()
                    plot_data['episodes'] = df['mean_episodes'].tolist()
                    plot_data['mean_checkpoints_per_episode'] = df['mean_mean_checkpoints_per_episode'].tolist()
                else:
                    sb3 = True
                    plot_data['mean_episode_rewards'] = experiments[experiment]['data']['mean_episode_rewards']

            else:
                plot_data['cum_rewards'], plot_data['mean_episode_rewards'], plot_data['smoothed_rewards'], plot_data[
                    'mean_steps_per_episode'], plot_data['episodes'], plot_data[
                    'mean_checkpoints_per_episode'] = calculate_values_from_data_for_plots(
                    experiments[experiment]["rewards_per_rollout"].values(),
                    experiments[experiment]["timesteps_per_episode_per_rollout"].values(),
                    experiments[experiment]["checkpoints_passed_per_episode_per_rollout"].values())

            if "exp-28" in label or "random" in label:
                for key in plot_data:
                    plot_data[key] = list(np.tile(sum(plot_data[key]) / len(plot_data[key]), 250))

            if not sb3:
                y_labels = ['Cumulated Episode Rewards', 'Mean Episode Rewards', 'Smoothed Rewards',
                            'Mean Steps / Episode',
                            'Number of Episodes', 'Mean Checkpoints / Episode']
                plot_line(axs, plot_data, label, y_labels, rollouts, n_steps, fontsize)

            else:
                max_number_of_x_labels = 10
                x_label = "Total Timesteps \n (Rollouts)"

                # calculate the array that maps the rollouts to total timesteps
                rollout_range = range(1, rollouts + 1)
                total_timestep_ticks = [i * n_steps for i in rollout_range]

                # calculate the ticks (= rollouts) that should be shown
                interval = len(total_timestep_ticks) / max_number_of_x_labels
                ticks_to_show = [i for i in rollout_range if (i + 1) % interval == 0]

                # calculate the ticks that should be used as indices
                xticks = [i - 1 for i in ticks_to_show]

                # create the labels that should be finally shown
                xticklabels = [f"{i * n_steps} \n ({i})" for i in ticks_to_show]
                axs.plot(plot_data['mean_episode_rewards'], label=label)
                axs.set_xticklabels(xticklabels)
                axs.set_xticks(xticks)
                axs.set_xlabel(x_label, fontsize=fontsize)
                axs.set_ylabel('Mean Episode Rewards', fontsize=fontsize)
                axs.legend()
                axs.grid(True)

        plt.tight_layout()
        fig.savefig(fig_name + ".png")


def _calculate_mean_from_data(sb3, mean_name, data):
    """
    calculates the mean of given data
    """
    keys = list(data.keys())
    len_keys = len(keys)

    if not sb3:
        if len_keys == 3:
            mean_cum_rewards = [(a + b + c) / len_keys for a, b, c in
                                zip_longest(*[data[keys[i]]['cum_rewards'] for i in range(len_keys)])]
            mean_mean_episode_rewards = [(a + b + c) / len_keys for a, b, c in
                                         zip_longest(*[data[keys[i]]['mean_episode_rewards'] for i in range(len_keys)])]
            mean_smoothed_rewards = [(a + b + c) / len_keys for a, b, c in
                                     zip_longest(*[data[keys[i]]['smoothed_rewards'] for i in range(len_keys)])]
            mean_mean_steps_per_episode = [(a + b + c) / len_keys for a, b, c in zip_longest(
                *[data[keys[i]]['mean_steps_per_episode'] for i in range(len_keys)])]
            mean_episodes = [(a + b + c) / len_keys for a, b, c in
                             zip_longest(*[data[keys[i]]['episodes'] for i in range(len_keys)])]
            mean_mean_checkpoints_per_episode = [(a + b + c) / len_keys for a, b, c in zip_longest(
                *[data[keys[i]]['mean_checkpoints_per_episode'] for i in range(len_keys)])]

        if len_keys == 2:
            mean_cum_rewards = [(a + b) / len_keys for a, b in
                                zip_longest(*[data[keys[i]]['cum_rewards'] for i in range(len_keys)])]
            mean_mean_episode_rewards = [(a + b) / len_keys for a, b in
                                         zip_longest(*[data[keys[i]]['mean_episode_rewards'] for i in range(len_keys)])]
            mean_smoothed_rewards = [(a + b) / len_keys for a, b in
                                     zip_longest(*[data[keys[i]]['smoothed_rewards'] for i in range(len_keys)])]
            mean_mean_steps_per_episode = [(a + b) / len_keys for a, b in zip_longest(
                *[data[keys[i]]['mean_steps_per_episode'] for i in range(len_keys)])]
            mean_episodes = [(a + b) / len_keys for a, b in
                             zip_longest(*[data[keys[i]]['episodes'] for i in range(len_keys)])]
            mean_mean_checkpoints_per_episode = [(a + b) / len_keys for a, b in zip_longest(
                *[data[keys[i]]['mean_checkpoints_per_episode'] for i in range(len_keys)])]

        data = {'mean_cum_rewards': mean_cum_rewards, 'mean_mean_episode_rewards': mean_mean_episode_rewards,
                'mean_smoothed_rewards': mean_smoothed_rewards,
                'mean_mean_steps_per_episode': mean_mean_steps_per_episode, 'mean_episodes': mean_episodes,
                'mean_mean_checkpoints_per_episode': mean_mean_checkpoints_per_episode}

    else:
        mean_episode_rewards = [(a + b) / len_keys for a, b in
                                zip_longest(*[data[keys[i]]['mean_episode_rewards'] for i in range(len_keys)])]
        data = {'mean_episode_rewards': mean_episode_rewards}

    df = pd.DataFrame.from_dict(data)
    df.to_csv("./scripts/" + mean_name + ".csv")


def _load_data(paths):
    experiments = {}
    sb3 = False

    for experiment in paths:

        path = os.path.join(*experiment.split("/")[0:-1]) if "mean" not in str(experiment) else str(experiment)
        rollouts = 0
        n_steps = 0
        seed = 0
        data = {"actor_loss_per_rollout": {}, "critic_loss_per_rollout": {}, "rewards_per_rollout": {},
                "timesteps_per_episode_per_rollout": {},
                "checkpoints_passed_per_episode_per_rollout": {}, "action_distribution": {}}

        if "mean" not in path:
            with open(experiment, "r") as f:
                yml = yaml.safe_load(f)
                rollouts = yml["agent"]["rollouts"]
                n_steps = yml["agent"]["model"]["parameters"]["n_steps"]
                seed = yml["environment"]["seed"]

        label = path.split('\\')[3] + "_" + str(seed) if "mean" not in str(path) else path.split('/')[2][:-4]

        if "sb3" in str(experiment):
            df = pd.read_csv(path + '/stats/progress.csv',
                             usecols=['time/total_timesteps', 'rollout/ep_rew_mean'])
            sb3_timesteps = df['time/total_timesteps'].tolist()
            sb3_episode_reward_mean = df['rollout/ep_rew_mean'].tolist()
            experiments.update({label: {'timesteps': sb3_timesteps, 'episode_reward_mean': sb3_episode_reward_mean,
                                        'rollouts': 250, 'n_steps': 4096}})
            sb3 = True
            continue

        if "mean" in str(experiment):
            df = pd.read_csv(path)
            experiments.update({label: {'data': df, 'rollouts': 250, 'n_steps': 4096}})
            if "sb" in experiment:
                sb3 = True
            continue

        path = os.path.join(path, "stats")
        for key in data.keys():
            e = os.path.join(path, key + ".json")
            with open(e, "r") as f:
                _data_dict = json.load(f)
                data[key] = eval(_data_dict)
        data.update({'rollouts': rollouts, 'n_steps': n_steps}) if not "random" in experiment else data.update(
            {'rollouts': 250, 'n_steps': 4096})
        experiments.update({label: data})

    return experiments, sb3


if __name__ == '__main__':
    main()
