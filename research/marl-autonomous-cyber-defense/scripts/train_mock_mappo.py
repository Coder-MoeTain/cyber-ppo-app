"""Short MAPPO smoke run on mock environment; not a CybORG benchmark."""
import argparse,json,math
from autocd.algorithms.mappo_trainer import MAPPOConfig,MAPPOTrainer
from autocd.envs.mock_enterprise import MockEnterpriseEnv
def main():
 p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=256);p.add_argument("--rollout-steps",type=int,default=64);p.add_argument("--seed",type=int,default=42);a=p.parse_args()
 trainer=MAPPOTrainer(MockEnterpriseEnv(seed=a.seed),MAPPOConfig(),seed=a.seed);history=trainer.train(a.steps,a.rollout_steps)
 for row in history:
  for k in ("actor_loss","critic_loss","entropy"):
   if not math.isfinite(row[k]):raise RuntimeError("Non-finite metric: "+k)
 print(json.dumps({"environment":"mock_only","algorithm":"MAPPO","seed":a.seed,"history":history},indent=2))
 print("WARNING: mock smoke test only; not CybORG performance evidence.")
if __name__=="__main__":main()
