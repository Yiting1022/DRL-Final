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

set_random_seed(42)

from torch.utils.tensorboard import SummaryWriter

class EpisodeStatsTensorboardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode_count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if info.get("episode_total_reward") is not None:
                self.episode_count += 1
                self.writer.add_scalar("episode_total_reward", info["episode_total_reward"], self.episode_count)
                if "final_stocks" in info:
                    self.writer.add_scalar("bot_final_stock", info["final_stocks"]["bot"], self.episode_count)
                    self.writer.add_scalar("cpu_final_stock", info["final_stocks"]["cpu"], self.episode_count)
        return True

    def _on_training_end(self):
        self.writer.close()




models_dir = os.path.join("models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(models_dir, exist_ok=True)

import glob
checkpoint_callback = CheckpointCallback(
    save_freq=1000000,  
    save_path=models_dir,
    name_prefix="ppo_melee_checkpoint",
    save_replay_buffer=True,
    verbose=1
)

reward_callback = EpisodeStatsTensorboardCallback(log_dir="./tensorboard_logs/")

device = "cpu"
print(f"Using device: {device}")    

env = DummyVecEnv([lambda: MeleeEnv()])

from stable_baselines3 import PPO
latest_checkpoint = None
if latest_checkpoint:
    print(f"Loading checkpoint: {latest_checkpoint}")
    model = PPO.load(latest_checkpoint, env=env, device=device, tensorboard_log="./tensorboard_logs/", verbose=1)
else:
    print("No checkpoint found, creating new model.")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=device,
        tensorboard_log="./tensorboard_logs/"
    )

print("Starting training...")
try:
    model.learn(
        total_timesteps=100000000,
        callback=[checkpoint_callback, reward_callback]
    )
    final_model_path = os.path.join(models_dir, "ppo_melee_final")
    model.save(final_model_path)
    print(f"Training completed. Model saved to {final_model_path}")
except KeyboardInterrupt:
    print("Training interrupted by user")
    interrupted_model_path = os.path.join(models_dir, "ppo_melee_interrupted")
    model.save(interrupted_model_path)
    print(f"Partially trained model saved to {interrupted_model_path}")
except Exception as e:
    print(f"Error during training: {e}")
finally:
    env.close()
    print("Environment closed")
