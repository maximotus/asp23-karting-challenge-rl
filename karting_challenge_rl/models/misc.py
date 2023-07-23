import torch


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def calculate_returns(rewards, values, discount_factor, dones, normalize=True):
    returns = []

    for r in reversed(range(len(rewards))):
        d = rewards[r] + discount_factor * values[r + 1] * dones[r] - values[r]
        R = d + discount_factor * 0.95 * dones[r] * d
        returns.insert(0, R + values[r])

    returns = torch.tensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def prepare_action_stats(action_list):
    action_stats = []
    for i in range(0, len(action_list[0])):
        action_stats.append([item[i] for item in action_list])
        action_stats[i] = sum(action_stats[i]) / len(action_stats[i])

    return action_stats
