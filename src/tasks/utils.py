import math


def scale_prob_reward_log(p, max_value, inverse=False):
    if inverse:
        p = 1 - p

    p = min(max(p, 1e-10), 1.0)
    if p <= 0:
        return 0  # Impossible
    if p >= 1:
        return max_value  # Certain

    logit = math.log(p / (1 - p))

    # Clamp logit values to avoid extreme outputs
    min_logit = -7
    max_logit = 7
    logit = max(-7, min(7, logit))

    # Normalize and scale to [0, max_value]
    scaled = (logit - min_logit) / (max_logit - min_logit) * max_value
    return scaled


def scale_logprob_reward_log(logprob, max_value, inverse=False):
    p = math.exp(logprob)
    return scale_prob_reward_log(p, max_value, inverse=inverse)

def scale_logprob_reward_linear(logprob, max_value, inverse=False):
    p = math.exp(logprob)
    return scale_prob_reward_linear(p, max_value, inverse=inverse)

def scale_prob_reward_linear(p, max_value, inverse=False):
    if inverse:
        p = 1 - p
    return p * max_value
