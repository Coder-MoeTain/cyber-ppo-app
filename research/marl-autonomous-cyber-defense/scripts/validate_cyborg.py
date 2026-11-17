"""Validate CC4 import, reset, observations, action masks and optional Sleep step."""
import argparse, json
import numpy as np
from autocd.envs.cyborg_adapter import CybORGAdapter

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=42,help="Recorded seed; actual simulator seeding must be verified")
    parser.add_argument("--one-step",action="store_true")
    args=parser.parse_args()
    env=CybORGAdapter(seed=args.seed)
    try:
        obs,infos=env.reset(seed=args.seed)
        report={"agents":sorted(obs),"observation_dims":env.observation_dims,"action_dims":env.action_dims,
                "valid_action_counts":{a:int(m.sum()) for a,m in env.action_masks().items()}}
        if set(obs)!=set(env.EXPECTED_AGENTS): raise RuntimeError(f"Expected five Blue agents, got {sorted(obs)}")
        if args.one_step:
            actions={}
            for a in env.agents:
                labels=env.env.action_labels(a)
                idx=next((i for i,x in enumerate(labels) if "sleep" in str(x).lower()),None)
                if idx is None: idx=int(np.flatnonzero(env.action_masks()[a])[0])
                actions[a]=idx
            _,rewards,terminated,truncated,_=env.step(actions)
            report["one_step"]={"rewards":rewards,"terminated":terminated,"truncated":truncated}
        print(json.dumps(report,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
    finally: env.close()
if __name__=="__main__": main()
