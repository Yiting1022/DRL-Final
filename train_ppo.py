#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from env import MeleeEnv, ACTION_SPACE
from utils import SkipFrame
import datetime
from torch.utils.tensorboard import SummaryWriter

# Update hyperparameters
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 2048      # rollout batch size
SEQUENCE_LENGTH = BATCH_SIZE
MINI_BATCH_SIZE = 256  # mini-batch size for updates
PPO_EPOCHS = 10        # update epochs
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
GAE_LAMBDA = 0.95
TOTAL_TIMESTEPS = 1000000000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_dir = os.path.join("tensorboard_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

models_dir = os.path.join("models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(models_dir, exist_ok=True)

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.policy_head = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

class Trajectory:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs)
        self.actions.append(action.item() if isinstance(action, torch.Tensor) else action)
        self.log_probs.append(log_prob.cpu() if isinstance(log_prob, torch.Tensor) else log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.cpu().item() if isinstance(value, torch.Tensor) else value)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

def compute_gae(rewards, values, dones, gamma, lam=GAE_LAMBDA):
    values = values + [0.0]
    gae = 0
    returns = []
    advantages = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
    return returns, advantages

def ppo_update(policy, optimizer, traj, gamma, clip_eps, vf_coef, ent_coef, global_step):
    returns, advantages = compute_gae(traj.rewards, traj.values, traj.dones, gamma)
    obs_tensor = torch.tensor(np.array(traj.obs), dtype=torch.float32, device=DEVICE)
    actions_tensor = torch.tensor(traj.actions, dtype=torch.long, device=DEVICE)
    old_log_probs_tensor = torch.stack(traj.log_probs).to(DEVICE)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    writer.add_scalar("Training/MeanReturns", returns_tensor.mean().item(), global_step)

    dataset_size = obs_tensor.size(0)
    indices = np.arange(dataset_size)
    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            mb_idx = indices[start:end]
            mb_obs = obs_tensor[mb_idx]
            mb_actions = actions_tensor[mb_idx]
            mb_old_log_probs = old_log_probs_tensor[mb_idx]
            mb_returns = returns_tensor[mb_idx]
            mb_advantages = advantages_tensor[mb_idx]
            
            logits, values = policy(mb_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_returns - values).pow(2).mean()
            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

def main():
    env = MeleeEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PPOActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    traj = Trajectory()

    obs, info = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    episode_reward = 0
    episode = 0
    step_count = 0
    update_count = 0
    episode_bot_stock_history = []
    episode_cpu_stock_history = []
    
    writer.add_text("Hyperparameters/LearningRate", str(LR), 0)
    writer.add_text("Hyperparameters/Gamma", str(GAMMA), 0)
    writer.add_text("Hyperparameters/ClipEps", str(CLIP_EPS), 0)
    writer.add_text("Hyperparameters/EntropyCoef", str(ENTROPY_COEF), 0)
    writer.add_text("Hyperparameters/ValueCoef", str(VF_COEF), 0)
    writer.add_text("Hyperparameters/BatchSize", str(BATCH_SIZE), 0)
    writer.add_text("Hyperparameters/SequenceLength", str(SEQUENCE_LENGTH), 0)
    writer.add_text("Environment/ActionSpace", str(len(ACTION_SPACE)), 0)
    writer.add_text("Environment/ObservationSpace", str(obs_dim), 0)
    writer.add_text("Model/Architecture", "MLP", 0)
    writer.add_text("Model/HiddenSize", str(256), 0)
    
    while step_count < TOTAL_TIMESTEPS:
        for t in range(BATCH_SIZE):
            with torch.no_grad():
                action, log_prob, value = policy.act(obs_tensor)
            action_idx = action.item()
            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            
            
            if "bot_stocks" in info and "cpu_stocks" in info:
                bot_stocks = info["bot_stocks"]
                cpu_stocks = info["cpu_stocks"]
                writer.add_scalar("Game/BotStocks", bot_stocks, step_count)
                writer.add_scalar("Game/CPUStocks", cpu_stocks, step_count)
                
                episode_bot_stock_history.append(bot_stocks)
                episode_cpu_stock_history.append(cpu_stocks)
            
            traj.add(obs_tensor, action, log_prob, reward, done, value)
            
            # 更新狀態
            obs = next_obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            episode_reward += reward
            step_count += 1
            
            if done:
                obs, info = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                episode += 1
                print(f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {step_count}")
                
                # 記錄episode統計數據到TensorBoard
                writer.add_scalar("Performance/EpisodeReward", episode_reward, episode)
                writer.add_scalar("Performance/EpisodeLength", t + 1, episode)
                
            
                if episode_bot_stock_history and episode_cpu_stock_history:
                    writer.add_scalar("Performance/FinalBotStock", episode_bot_stock_history[-1], episode)
                    writer.add_scalar("Performance/FinalCPUStock", episode_cpu_stock_history[-1], episode)
                
                # 計算勝負情況 (根據最終stocks)
                if episode_bot_stock_history and episode_cpu_stock_history:
                    final_bot_stock = episode_bot_stock_history[-1]
                    final_cpu_stock = episode_cpu_stock_history[-1]
                    
                    if final_bot_stock > final_cpu_stock:
                        writer.add_scalar("Performance/GameResult", 1, episode)  # 1 表示勝利
                    elif final_bot_stock < final_cpu_stock:
                        writer.add_scalar("Performance/GameResult", -1, episode)  # -1 表示失敗
                    else:
                        writer.add_scalar("Performance/GameResult", 0, episode)  # 0 表示平局
                
                # 記錄episode獎勵分佈
                writer.add_histogram("Training/EpisodeRewardDistribution", np.array([episode_reward]), episode)
                
                episode_reward = 0
                
                # 定期保存模型
                if episode % 50 == 0:
                    model_path = os.path.join(models_dir, f"ppo_mlp_melee_{episode}.pt")
                    torch.save({
                        'episode': episode,
                        'step_count': step_count,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, model_path)
                    print(f"模型已保存: {model_path}")
        
        # PPO更新
        if len(traj.obs) > SEQUENCE_LENGTH:
            update_count += 1
            ppo_update(policy, optimizer, traj, GAMMA, CLIP_EPS, VF_COEF, ENTROPY_COEF, update_count)
            traj.clear()
    
    # 訓練結束，保存最終模型
    final_model_path = os.path.join(models_dir, "ppo_mlp_final.pt")
    torch.save({
        'episode': episode,
        'step_count': step_count,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    print(f"訓練完成。最終模型已保存至: {final_model_path}")
    
    # 關閉TensorBoard writer
    writer.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n訓練被中斷，正在儲存最終模型...")
        writer.close()
    except Exception as e:
        print(f"訓練發生錯誤: {e}")
        writer.close()