import torch

def clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=.2):
    ratios = torch.exp(new_log_probs-old_log_probs)
    unclipped = ratios*advantages
    clipped = torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon)*advantages
    return -torch.minimum(unclipped, clipped).mean()

def value_loss(values, returns):
    return .5*(values-returns).pow(2).mean()

def normalize_advantages(advantages, eps=1e-8):
    return (advantages-advantages.mean())/(advantages.std(unbiased=False)+eps)

def compute_gae(rewards, values, dones, gamma=.99, gae_lambda=.95):
    if not (len(rewards)==len(values)==len(dones)):
        raise ValueError("rewards, values, dones lengths must match")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros((), dtype=values.dtype, device=values.device)
    for t in reversed(range(len(rewards))):
        nonterminal = 1.-dones[t].float()
        delta = rewards[t]+gamma*next_value*nonterminal-values[t]
        gae = delta+gamma*gae_lambda*nonterminal*gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages+values
