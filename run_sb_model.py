#!/usr/bin/env python3

import os
from env import MeleeEnv, ACTION_SPACE  # Import ACTION_SPACE from env.py
from stable_baselines3 import PPO
import numpy as np
import time
from utils import SkipFrame  # Assuming you have a SkipFrame utility for frame skipping

# Manual configuration (replace with your values)
MODEL_PATH = "models/20250531-221107/ppo_melee_interrupted"  # Path to the trained model
EPISODES = 10          # Number of episodes to run
DETERMINISTIC = True   # Whether to use deterministic actions
SEED = 42              # Random seed (None for no seed)

def main():
    # Set random seed if provided
    if SEED is not None:
        np.random.seed(SEED)
    
    # Create environment
    env = MeleeEnv()
    env  = SkipFrame(env, skip=3)
    
    # Load the trained model
    print(f"Loading model from {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    
    # Statistics
    total_rewards = []
    win_count = 0
    loss_count = 0
    draw_count = 0
    
    print(f"Running for {EPISODES} episodes...")
    
    for episode in range(EPISODES):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        print(f"Episode {episode+1}/{EPISODES}")
        print(f"Playing as {info['bot_character']} vs {info['cpu_character']} (Level {info['cpu_level']}) on {info['stage']}")
        
        while not done and not truncated:
            # Get action from model (returns an integer)
            action, _states = model.predict(obs, deterministic=DETERMINISTIC)
            
            # Convert integer action to the format expected by the environment
            # The model outputs an index into the ACTION_SPACE array
            action_index = action.item() if hasattr(action, 'item') else int(action)
            action_tuple = ACTION_SPACE[action_index]
            
            # Take step with the properly formatted action
            obs, reward, done, truncated, info = env.step(action_tuple)
            episode_reward += reward
            step_count += 1
            
            if step_count % 100 == 0:  # Print status every 100 steps
                print(f"Step: {step_count}, Current reward: {episode_reward:.2f}, Bot stocks: {info['bot_stocks']}, CPU stocks: {info['cpu_stocks']}")
        
        # Game over
        total_rewards.append(episode_reward)
        
        # Track win/loss
        if info.get('game_over_flag') == 'win':
            win_count += 1
            result = "WIN"
        elif info.get('game_over_flag') == 'lose':
            loss_count += 1
            result = "LOSE"
        elif info.get('game_over_flag') == 'draw':
            draw_count += 1
            result = "DRAW"
        else:
            result = "UNKNOWN"
            
        print(f"Episode {episode+1} finished - Result: {result}, Total reward: {episode_reward:.2f}")
        print(f"Statistics: {info}")
        print("-" * 50)
        
        # Small delay between episodes
        time.sleep(1)
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("Run Complete!")
    print(f"Episodes: {EPISODES}")
    print(f"Win rate: {win_count/EPISODES:.2%} ({win_count} wins, {loss_count} losses, {draw_count} draws)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print("=" * 50)
    
    env.close()

if __name__ == "__main__":
    main()
