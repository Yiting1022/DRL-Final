#!/usr/bin/env python3

import os
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from env import MeleeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # 🟢 改用 DummyVecEnv
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from utils import SkipFrame  # Assuming you have a SkipFrame utility for frame skipping

set_random_seed(42)

class EpisodeStatsTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if info.get("episode_total_reward") is not None:
                self.episode_count += 1
                # 使用 episode_count 作為 step
                self.logger.record("episode_total_reward", info["episode_total_reward"], self.episode_count)
                if "final_stocks" in info:
                    self.logger.record("bot_final_stock", info["final_stocks"]["bot"], self.episode_count)
                    self.logger.record("cpu_final_stock", info["final_stocks"]["cpu"], self.episode_count)
        return True

if __name__ == "__main__":
    models_dir = os.path.join("models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(models_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=1000000,  
        save_path=models_dir,
        name_prefix="ppo_melee_checkpoint",
        save_replay_buffer=True,
        verbose=1
    )

    reward_callback = EpisodeStatsTensorboardCallback()

    device = "cpu"
    print(f"Using device: {device}")    

    NUM_ENVS = 1
    def make_env(rank):
        def _init():
            env = MeleeEnv(slippi_port=51298 + rank*10)
            return env
        return _init

    env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])

    latest_checkpoint = "models/20250604-004625/ppo_melee_interrupted"  # Replace with your actual checkpoint path
    if latest_checkpoint:
        print(f"Loading checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, device=device, tensorboard_log="./tensorboard_logs/", verbose=0)
    else:
        print("No checkpoint found, creating new model.")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            device=device,
            tensorboard_log="./tensorboard_logs/",
            gamma=0.99,
            n_steps=2048,
            gae_lambda=0.95, 
            ent_coef=0.01,   
            clip_range=0.2,     
            batch_size=NUM_ENVS * 64  
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
