from fastapi import FastAPI
from CyberSecurityEnv import CyberSecurityEnv
from stable_baselines3 import PPO
import torch

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = CyberSecurityEnv()
model = PPO.load("model/enhanced_cyber_security_ppo_gpu", device=device)

@app.get("/")
def root():
    return {"message": "Cyber Security PPO Model is Running"}

@app.post("/predict/")
def predict():
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    return {"predicted_action": int(action)}
