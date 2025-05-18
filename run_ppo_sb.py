#!/usr/bin/env python3

import os
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from env import MeleeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

# 设置随机种子以确保可重复性
set_random_seed(42)

# 创建自定义回调函数来记录平均奖励
class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq=1000, log_dir=None, verbose=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []

    def _init_callback(self):
        if self.log_dir is not None:
            os.makedirs(os.path.join(self.log_dir, "rewards"), exist_ok=True)
            self.reward_file = open(os.path.join(self.log_dir, "rewards", "rewards.csv"), "w")
            self.reward_writer = csv.writer(self.reward_file)
            self.reward_writer.writerow(["Steps", "Average Reward", "Min Reward", "Max Reward"])

    def _on_step(self):
        # 收集每个环境的奖励
        for info in self.locals["infos"]:
            if "episode" in info:
                self.current_rewards.append(info["episode"]["r"])
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        
        # 每check_freq步，计算并记录平均奖励
        if self.n_calls % self.check_freq == 0 and len(self.current_rewards) > 0:
            mean_reward = np.mean(self.current_rewards)
            min_reward = np.min(self.current_rewards)
            max_reward = np.max(self.current_rewards)
            self.rewards.append((self.n_calls, mean_reward))
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Average reward: {mean_reward:.2f} (Min: {min_reward:.2f}, Max: {max_reward:.2f})")
            
            # 保存到CSV
            if self.log_dir is not None:
                self.reward_writer.writerow([self.n_calls, mean_reward, min_reward, max_reward])
                self.reward_file.flush()
            
            # 清空当前周期的奖励列表
            self.current_rewards = []
            
            # 绘制奖励曲线
            if len(self.rewards) > 1:
                steps, rewards = zip(*self.rewards)
                plt.figure(figsize=(10, 6))
                plt.plot(steps, rewards)
                plt.xlabel("Steps")
                plt.ylabel("Average Reward")
                plt.title("Training Average Reward Over Time")
                plt.savefig(os.path.join(self.log_dir, "rewards", "reward_plot.png"))
                plt.close()
        
        return True
    
    def _on_training_end(self):
        if self.log_dir is not None:
            self.reward_file.close()
            
            # 保存完整的训练统计信息
            with open(os.path.join(self.log_dir, "rewards", "training_stats.txt"), "w") as f:
                f.write(f"Total steps: {self.n_calls}\n")
                if len(self.episode_rewards) > 0:
                    f.write(f"Average episode reward: {np.mean(self.episode_rewards):.2f}\n")
                    f.write(f"Median episode reward: {np.median(self.episode_rewards):.2f}\n")
                    f.write(f"Min episode reward: {np.min(self.episode_rewards):.2f}\n")
                    f.write(f"Max episode reward: {np.max(self.episode_rewards):.2f}\n")
                    f.write(f"Standard deviation: {np.std(self.episode_rewards):.2f}\n")
                    f.write(f"Average episode length: {np.mean(self.episode_lengths):.2f}\n")
                    
                    # 保存完整的奖励历史
                    np.save(os.path.join(self.log_dir, "rewards", "episode_rewards.npy"), np.array(self.episode_rewards))

# 创建保存模型的目录
models_dir = os.path.join("models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(models_dir, exist_ok=True)

# 创建检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # 每10000步保存一次
    save_path=models_dir,
    name_prefix="ppo_melee_checkpoint",
    save_replay_buffer=True,
    verbose=1
)

# 创建奖励记录回调
reward_callback = RewardLoggerCallback(check_freq=1000, log_dir=models_dir, verbose=1)

device = "cpu"
print(f"Using device: {device}")    

env = DummyVecEnv([lambda: MeleeEnv()])
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    device=device,
    tensorboard_log="./tensorboard_logs/"  # 添加TensorBoard日志
)

print("Starting training...")
try:
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, reward_callback]  # 使用两个回调
    )
    final_model_path = os.path.join(models_dir, "ppo_melee_final")
    model.save(final_model_path)
    print(f"Training completed. Model saved to {final_model_path}")
except KeyboardInterrupt:
    print("Training interrupted by user")
    # 保存中断时的模型
    interrupted_model_path = os.path.join(models_dir, "ppo_melee_interrupted")
    model.save(interrupted_model_path)
    print(f"Partially trained model saved to {interrupted_model_path}")
except Exception as e:
    print(f"Error during training: {e}")
finally:
    env.close()
    print("Environment closed")
