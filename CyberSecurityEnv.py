import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch
from collections import deque
import random
import matplotlib.pyplot as plt

# Check GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Enhanced Data Loading and Preprocessing
def load_and_process_daily_files():
    daily_files = [
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv'
    ]
    
    dfs = []
    for file in daily_files:
        filepath = os.path.join('CIC-IDS2017', file)
        if os.path.exists(filepath):
            print(f"Loading {file}...")
            try:
                df = pd.read_csv(filepath, low_memory=False)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        else:
            print(f"File not found: {file}")
    
    if not dfs:
        print("No data files found")
        return None
    
    full_df = pd.concat(dfs, axis=0)
    full_df.columns = full_df.columns.str.strip()
    full_df = full_df.dropna(axis=1, how='all')
    
    # Enhanced feature selection
    cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Fwd Header Length']
    cols_to_drop = [col for col in cols_to_drop if col in full_df.columns]
    full_df = full_df.drop(cols_to_drop, axis=1)
    
    # Handle infinite values and NaNs
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns
    full_df[numeric_cols] = full_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    full_df = full_df.dropna()
    
    # Advanced label encoding
    label_encoder = LabelEncoder()
    full_df['Label'] = label_encoder.fit_transform(full_df['Label'])
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", label_mapping)
    
    # Smart class balancing with stratified sampling
    rus = RandomUnderSampler(random_state=42, sampling_strategy='not minority')
    X_res, y_res = rus.fit_resample(full_df.drop('Label', axis=1), full_df['Label'])
    balanced_df = pd.DataFrame(X_res, columns=full_df.drop('Label', axis=1).columns)
    balanced_df['Label'] = y_res
    
    balanced_df.to_csv('CIC-IDS2017/processed_cic_ids2017.csv', index=False)
    return balanced_df

# 2. Enhanced CyberSecurity Environment with GPU support
class CyberSecurityEnv(gym.Env):
    def __init__(self, data_path='CIC-IDS2017/processed_cic_ids2017.csv'):
        super(CyberSecurityEnv, self).__init__()
        
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        
        self.labels = self.data['Label'].values
        features = self.data.drop('Label', axis=1)
        
        # Enhanced feature scaling
        self.scaler.fit(features)
        self.features = self.scaler.transform(features).astype(np.float32)
        
        # Dynamic action space based on unique labels
        self.unique_labels = np.unique(self.labels)
        self.action_space = spaces.Discrete(len(self.unique_labels))
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.features.shape[1],), 
            dtype=np.float32
        )
        
        # Adaptive episode length
        self.episode_length = min(2000, len(self.data) // 5)
        self.current_step = 0
        self.start_step = 0
        self.attack_history = deque(maxlen=200)
        
    def reset(self, seed=None, options=None):
        max_start = max(1, len(self.data) - self.episode_length)
        self.current_step = random.randint(0, max_start)
        self.start_step = self.current_step
        self.attack_history.clear()
        obs = self._next_observation()
        return obs, {}
        
    def _next_observation(self):
        return self.features[self.current_step]
        
    def step(self, action):
        label = self.labels[self.current_step]
        reward = 0
        done = False
        info = {'label': label}
        
        # Enhanced reward structure
        if action == label:
            reward = 3.0  # Higher reward for correct classification
        else:
            if label == 0:  # False positive
                reward = -3.0
            else:  # False negative or wrong attack type
                if action == 0:  # Missed attack completely
                    reward = -5.0
                else:  # Wrong attack type
                    reward = -1.0
        
        self.current_step += 1
        
        # Termination conditions
        if self.current_step >= len(self.data) - 1:
            done = True
        elif self.current_step - self.start_step >= self.episode_length:
            done = True
            
        next_obs = self._next_observation()
        return next_obs, reward, done, False, info

# 3. GPU-accelerated Training Function
def train_agent():
    data = load_and_process_daily_files()
    if data is None:
        print("Data processing failed")
        return None
    
    env = CyberSecurityEnv()
    
    try:
        check_env(env)
        print("Environment check passed successfully")
    except Exception as e:
        print(f"Environment check failed: {str(e)}")
        return None
    
    # Early stopping callback
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_on_best,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Enhanced neural network architecture with GPU support
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
       net_arch=dict(pi=[1024, 512, 256, 128], vf=[1024, 512, 256, 128]),
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-4, eps=1e-5)
    )
    
    # Configure PPO for GPU training
    model = PPO(
        'MlpPolicy',
        env,
        device=device,  # Enable GPU acceleration
        verbose=2,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        target_kl=0.02,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("Starting training with GPU acceleration...")
    model.learn(
        total_timesteps=200000,
        callback=eval_callback,
        tb_log_name="ppo_cyber_gpu",
        progress_bar=True
    )
    
    model.save("enhanced_cyber_security_ppo_gpu")
    print("Training completed and model saved")
    return model

# 4. Comprehensive Evaluation Function
def evaluate_model(model, num_episodes=20):
    env = CyberSecurityEnv()
    
    results = {
        'total_rewards': [],
        'confusion_matrix': np.zeros((len(env.unique_labels), len(env.unique_labels))),
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total': 0
        }
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            
            # Update confusion matrix
            true_label = info['label']
            results['confusion_matrix'][true_label][action] += 1
            
            # Update episode stats
            if action == true_label:
                episode_stats['true_positives'] += 1
            else:
                if true_label == 0:
                    episode_stats['false_positives'] += 1
                else:
                    episode_stats['false_negatives'] += 1
            episode_stats['total'] += 1
        
        results['total_rewards'].append(total_reward)
        
        # Calculate precision, recall, f1 for this episode
        precision = episode_stats['true_positives'] / (episode_stats['true_positives'] + episode_stats['false_positives'] + 1e-6)
        recall = episode_stats['true_positives'] / (episode_stats['true_positives'] + episode_stats['false_negatives'] + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        
        print(f"Episode {ep+1}: Reward={total_reward:.1f}, Precision={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}")
    
    # Calculate overall metrics
    print("\n=== Final Evaluation Metrics ===")
    print(f"Average Reward: {np.mean(results['total_rewards']):.2f} ± {np.std(results['total_rewards']):.2f}")
    print(f"Average Precision: {np.mean(results['precision']):.2%} ± {np.std(results['precision']):.2%}")
    print(f"Average Recall: {np.mean(results['recall']):.2%} ± {np.std(results['recall']):.2%}")
    print(f"Average F1 Score: {np.mean(results['f1']):.2%} ± {np.std(results['f1']):.2%}")
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(results['confusion_matrix'], cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.show()
    
    return results

# 5. Main Execution
if __name__ == "__main__":
    # Train or load model
    try:
        model = PPO.load("enhanced_cyber_security_ppo_gpu", device=device)
        print("Loaded trained model")
    except:
        print("No saved model found, training new model...")
        model = train_agent()
    
    # Evaluate model
    if model is not None:
        results = evaluate_model(model, num_episodes=20)
        
        # Plot training rewards
        if os.path.exists("./tensorboard_logs/ppo_cyber_gpu"):
            from stable_baselines3.common.logger import read_csv
            try:
                log_df = read_csv("./tensorboard_logs/ppo_cyber_gpu")
                plt.figure(figsize=(10, 5))
                plt.plot(log_df['time/total_timesteps'], log_df['rollout/ep_rew_mean'])
                plt.title("Training Progress")
                plt.xlabel("Timesteps")
                plt.ylabel("Average Episode Reward")
                plt.grid(True)
                plt.show()
            except Exception as e:
                print(f"Could not plot training progress: {e}")