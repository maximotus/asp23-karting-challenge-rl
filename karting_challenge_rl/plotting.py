def calculate_values_from_data_for_plots(rewards_per_rollout_input, timesteps_per_episode_per_rollout_input, checkpoints_passed_per_episode_per_rollout_input):
    """
    calculates values for plots
    """
    smoothing_window_size = 10
    mean_episode_rewards_window = []
    cum_episodes = []
    total_number_of_episodes = 0
    # calculates values for 
    cum_rewards = []
    mean_episode_rewards = []
    smoothed_rewards = []
    mean_steps_per_episode = []
    episodes = []
    mean_checkpoints_per_episode = []

    # prepare the reward-related data that will be plotted
    for rewards_per_episode in rewards_per_rollout_input:
        cum_episode_rewards = []

        for rewards_per_episode in rewards_per_episode.values():
            cum_episode_rewards.append(sum(rewards_per_episode))
            
        cum_reward = sum(cum_episode_rewards)
        cum_rewards.append(cum_reward)
        mean_reward = cum_reward / len(cum_episode_rewards)
        mean_episode_rewards.append(mean_reward)
        mean_episode_rewards_window.append(mean_reward)
        if len(mean_episode_rewards_window) > smoothing_window_size:
            mean_episode_rewards_window.pop(0)
        mean_mean_reward = sum(mean_episode_rewards_window) / len(mean_episode_rewards_window)
        smoothed_rewards.append(mean_mean_reward)
    
    # prepare the episode-related data that will be plotted
    for timesteps_per_episodes in timesteps_per_episode_per_rollout_input:
        number_of_episodes = len(timesteps_per_episodes)
        mean_steps = sum(list(timesteps_per_episodes.values())) / number_of_episodes
        episodes.append(number_of_episodes)
        total_number_of_episodes += number_of_episodes
        cum_episodes.append(total_number_of_episodes)
        mean_steps_per_episode.append(mean_steps)

    # prepare the checkpoint data
    for checkpoints_passed_per_ep in checkpoints_passed_per_episode_per_rollout_input:
        number_of_episodes = len(checkpoints_passed_per_ep)
        mean_checkpoints = sum(list(checkpoints_passed_per_ep.values())) / number_of_episodes
        mean_checkpoints_per_episode.append(mean_checkpoints)

    return cum_rewards, mean_episode_rewards, smoothed_rewards, mean_steps_per_episode, episodes, mean_checkpoints_per_episode


def plot_line(axs, plot_data, label, y_labels, rollouts, n_steps, fontsize):
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

    for i, key in enumerate(plot_data.keys()):
        axs[i].plot(plot_data[key], label=label)
        axs[i].set_xticklabels(xticklabels)
        axs[i].set_xticks(xticks)
        axs[i].set_xlabel(x_label, fontsize=fontsize)
        axs[i].set_ylabel(y_labels[i], fontsize=fontsize)
        axs[i].legend()
        axs[i].grid(True)
