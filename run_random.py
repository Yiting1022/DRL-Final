#!/usr/bin/env python3

import random
import time
import sys

# Try to import gymnasium, if unavailable use old gym

import gymnasium as gym
from gymnasium import spaces
from utils import SkipFrame  # 新增這行
# Import the environment class
from env import MeleeEnv, ACTION_SPACE


def run_random_agent():
    """Run a random agent in the Melee environment"""
    env = MeleeEnv()
    # Track statistics
    episode_count = 0
    total_rewards = []
    episode_lengths = []
    
    try:
        while True:
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:

                action_idx = env.action_space.sample()
                action = ACTION_SPACE[action_idx]
                obs, reward, terminated, truncated, info  = env.step(action)
                done = terminated or truncated                
                total_reward += reward
                steps += 1
            
            episode_count += 1
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            avg_reward = sum(total_rewards) / len(total_rewards)
            avg_length = sum(episode_lengths) / len(episode_lengths)
            
            print(f"Episode {episode_count} finished with total reward: {total_reward:.2f} in {steps} steps")
            print(f"Average reward over {episode_count} episodes: {avg_reward:.2f}")
            print(f"Average episode length: {avg_length:.1f} steps")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    run_random_agent()
