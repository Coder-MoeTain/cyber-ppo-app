def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0

def summarize_episode(episode_infos):
    if not episode_infos:
        return {"availability": 0.0, "weighted_compromise": 0.0}
    return {
        "availability": sum(float(x.get("availability", 0.0)) for x in episode_infos) / len(episode_infos),
        "weighted_compromise": sum(float(x.get("weighted_compromise", 0.0)) for x in episode_infos) / len(episode_infos),
    }
