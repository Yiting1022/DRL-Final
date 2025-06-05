#!/usr/bin/env python3

import os
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from env import MeleeEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import traceback
import csv
import ipdb

# 设置随机种子以确保可重复性
set_random_seed(42)

class EpisodeStatsTensorboardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self):
        try:
            for info in self.locals.get("infos", []):
                if info.get("episode_total_reward") is not None:
                    self.episode_count += 1
                    self.logger.record("episode_total_reward", info["episode_total_reward"])
                    self.logger.record("episode_count", self.episode_count)
                    if "final_stocks" in info:
                        self.logger.record("bot_final_stock", info["final_stocks"]["bot"])
                        self.logger.record("cpu_final_stock", info["final_stocks"]["cpu"])
        except Exception as e:
            print(f"Error in callback _on_step: {e}")
            traceback.print_exc()
        return True

    def _on_training_end(self):
        return True

# 创建保存模型的目录
models_dir = os.path.join("models", "double_dueling_dqn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(models_dir, exist_ok=True)

# 创建检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # 每10000步保存一次
    save_path=models_dir,
    name_prefix="double_dueling_dqn_melee_checkpoint",
    save_replay_buffer=True,
    verbose=1
)

# 创建奖励记录回调
reward_callback = EpisodeStatsTensorboardCallback(log_dir=models_dir)

# Check if CUDA is available and use GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

env = DummyVecEnv([lambda: MeleeEnv()])

# Create custom policy kwargs for dueling network
policy_kwargs = dict(
    net_arch=[128, 128],  # Network architecture
    dueling=True,  # Enable dueling architecture
)

model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    device=device,
    tensorboard_log="./tensorboard_logs/",
    policy_kwargs=policy_kwargs,
    # Double DQN is enabled by default in stable-baselines3
    target_update_interval=1000,
)

print("Starting Double Dueling DQN training...")
try:
    model.learn(
        total_timesteps=20000000,
        callback=[checkpoint_callback, reward_callback]
    )
    final_model_path = os.path.join(models_dir, "double_dueling_dqn_melee_final")
    model.save(final_model_path)
    print(f"Double Dueling DQN training completed. Model saved to {final_model_path}")
except KeyboardInterrupt:
    print("Training interrupted by user")
    interrupted_model_path = os.path.join(models_dir, "double_dueling_dqn_melee_interrupted")
    model.save(interrupted_model_path)
    print(f"Partially trained model saved to {interrupted_model_path}")
except Exception as e:
    print(f"Error during training: {e}")
finally:
    env.close()
    print("Environment closed")
