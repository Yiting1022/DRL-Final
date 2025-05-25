#!/usr/bin/env python3

import melee
from melee import enums, Menu, MenuHelper
import numpy as np
import random
import os
import sys
import time
from config import CONFIG
from state import form_state, calculate_reward, get_distance
debug = False

import gymnasium as gym
from gymnasium import spaces
gym_module = gym
env_class = gym.Env

cardinal_sticks = [
    (0, 0.5),
    (1, 0.5),
    (0.5, 0),
    (0.5, 1),
    (0.5, 0.5)
]
tilt_sticks = [
    (0.4, 0.5),
    (0.6, 0.5),
    (0.5, 0.4),
    (0.5, 0.6)
]
diagonal_sticks = [
    (0, 0), (0, 0.5), (0, 1),
    (0.5, 0), (0.5, 0.5), (0.5, 1),
    (1, 0), (1, 0.5), (1, 1)
]
neutral_stick = [(0.5, 0.5)]

SimpleButton = enums.Button

ACTION_SPACE = []
for btn in [SimpleButton.BUTTON_A, SimpleButton.BUTTON_B]:
    for stick in cardinal_sticks:
        ACTION_SPACE.append((btn, stick))
for stick in tilt_sticks:
    ACTION_SPACE.append((SimpleButton.BUTTON_A, stick))
for btn in [None, SimpleButton.BUTTON_L]:
    for stick in diagonal_sticks:
        ACTION_SPACE.append((btn, stick))
for btn in [SimpleButton.BUTTON_Z, SimpleButton.BUTTON_Y]:
    for stick in neutral_stick:
        ACTION_SPACE.append((btn, stick))


class MeleeEnv(env_class):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config=None):
        self.total_reward = 0
        if gym_module is not None:
            super().__init__()
            
        self.config = config if config is not None else CONFIG
        
        if spaces is not None:
            self.action_space = spaces.Discrete(len(ACTION_SPACE))
            self.observation_space = spaces.Box(
                low=-float('inf'), 
                high=float('inf'), 
                shape=(46,), 
                dtype=np.float32
            )
        
        console_params = {
            "path": self.config["dolphin_path"],
            "slippi_address": self.config["slippi_address"],
            "slippi_port": self.config["slippi_port"]
        }
        
        if "gfx_backend" in self.config:
            console_params["gfx_backend"] = self.config["gfx_backend"]
        if "disable_audio" in self.config:
            console_params["disable_audio"] = self.config["disable_audio"]
        if "use_exi_inputs" in self.config:
            console_params["use_exi_inputs"] = self.config["use_exi_inputs"]
        if "enable_ffw" in self.config:
            console_params["enable_ffw"] = self.config["enable_ffw"]
        
        self.console = melee.Console(**console_params)

        self.console.run(iso_path=self.config["iso_path"])

        self.ports = self.config["ports"]
        self.characters = self.config["characters"]
        self.cur_char = []
        self.agent_levels = self.config["levels"]
        self.cur_level = []
        self.costumes = self.config["costumes"]
        self.stages = self.config["stages"]
        self.ctrls = []
        

        for i in range(len(self.ports)):
            idx = i if i < len(self.characters) else 0
            self.cur_char.append(random.choice(self.characters[idx]))
            self.cur_level.append(random.choice(self.agent_levels[idx]))
            
        for port in self.ports:
            self.ctrls.append(melee.Controller(console=self.console, port=port, type=melee.ControllerType.STANDARD))
        
        if not self.console.connect():
            raise RuntimeError("Failed to connect to Dolphin.")
        
        for ctrl in self.ctrls:  
            if not ctrl.connect():
                raise RuntimeError(f"Failed to connect controller on port {ctrl.port}.")
        self.stage = random.choice(self.stages)
        self.current_action = None
        self.frame_count = 0
        self.game_over_flag = None
        self.prev_gamestate = None  
        self.menu_helper = MenuHelper()
        self.in_game_frame_counter = 0

    def reset(self, seed=None, options=None):
        self.total_reward = 0
        gs = self.console.step()
        print(f"DEBUG: Resetting game state: {gs.menu_state}")
        if seed is not None and hasattr(gym, 'Env'):
            random.seed(seed)
            np.random.seed(seed)

        for i in range(len(self.ports)):
            self.cur_char[i] = random.choice(self.characters[i])
            self.cur_level[i] = random.choice(self.agent_levels[i])
            print(f"DEBUG: Player {i} character: {self.cur_char[i]}, level: {self.cur_level[i]}")
        
        self.game_over_flag = None
        last_menu_state = None
        frame_counter = 0 
        self.stage = random.choice(self.stages) 
        # release all first
        for ctrl in self.ctrls:
            ctrl.release_all()
        
        while True:
            gs = self.console.step()
            if gs is None:
                continue

            frame_counter += 1

            state = gs.menu_state
            if state != last_menu_state:
                print(f'last_menu_state: {last_menu_state}, current_menu_state: {state}')   
                if state == Menu.MAIN_MENU or state == Menu.UNKNOWN_MENU:
                    if debug:   
                        print("DEBUG: Main Menu")
                elif state == Menu.CHARACTER_SELECT:
                    if debug:   
                        print(f"DEBUG: Choosing characters: {self.cur_char}")
                    self.menu_helper.name_tag_index = 0
                    self.menu_helper.inputs_live = False
                elif state == Menu.STAGE_SELECT:
                    if debug:
                        print(f"DEBUG: Choosing stage: {self.stage}")
                    self.menu_helper.stage_selected = False
                elif state in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
                    if debug:
                        print("DEBUG: In Game")
                last_menu_state = state

            if state == Menu.MAIN_MENU or state == Menu.UNKNOWN_MENU:
                self.menu_helper.choose_versus_mode(
                    gamestate=gs,
                    controller=self.ctrls[0]
                )
            elif state == Menu.CHARACTER_SELECT:
                if hasattr(gs, "frame") and gs.frame < 20:
                    for ctrl in self.ctrls:
                        ctrl.release_all()
                else:
                    
                    self.menu_helper.choose_character(
                        gamestate=gs,
                        controller=self.ctrls[1],
                        character=self.cur_char[1],
                        cpu_level=self.cur_level[1],
                        costume=self.costumes[1],   
                        swag=False,
                        start=True
                    )
                    self.menu_helper.choose_character(
                        gamestate=gs,
                        controller=self.ctrls[0],
                        character=self.cur_char[0],
                        cpu_level=self.cur_level[0],
                        costume=self.costumes[0],
                        swag=False,
                        start=False
                    )
            elif state == Menu.STAGE_SELECT:
                self.menu_helper.choose_stage(
                    gamestate=gs,
                    controller=self.ctrls[0],
                    stage=self.stage,
                    character=self.cur_char[0],
                )
            elif state in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
                self.in_game_frame_counter = 0  
                self.prev_gamestate = gs
                observation = form_state(gs, self.ports[0], self.ports[1], stage_mode=1)
                
                info = {
                    "stage": gs.stage.name if hasattr(gs.stage, "name") else str(gs.stage),
                    "bot_character": gs.players[self.ports[0]].character.name if hasattr(gs.players[self.ports[0]].character, "name") else str(gs.players[self.ports[0]].character),
                    "cpu_character": gs.players[self.ports[1]].character.name if hasattr(gs.players[self.ports[1]].character, "name") else str(gs.players[self.ports[1]].character),
                    "cpu_level": gs.players[self.ports[1]].cpu_level,
                    "frame": 0
                }
                return observation, info
         
        
                
    def step(self, action):
        if isinstance(action, (int, np.integer)) and hasattr(self, 'action_space'):
            action = ACTION_SPACE[action]
            
        button, direction = action
        if button is not None:
            self.ctrls[0].press_button(button)
        self.ctrls[0].tilt_analog(enums.Button.BUTTON_MAIN, direction[0], direction[1])
        self.ctrls[0].flush()
        
       
        prev_gs = self.prev_gamestate
        next_gs = self.console.step()
        if next_gs is None:
            info = {"warning": "Received None gamestate"}
            return None, 0.0, False, False, info

        if next_gs.menu_state in (Menu.CHARACTER_SELECT, Menu.STAGE_SELECT):
            observation = None 
            reward = 0.0
            info = {
                "terminal_reason": "menu_exit",
                "terminal_observation": None
            }

            return observation, reward, True, False, info


        if self.game_over_flag is None:  
            bot_stock = next_gs.players[self.ports[0]].stock
            cpu_stock = next_gs.players[self.ports[1]].stock
            if bot_stock == 0 and cpu_stock == 0:
                self.game_over_flag = "draw"
            elif cpu_stock == 0:
                self.game_over_flag = "win"
            elif bot_stock == 0:
                self.game_over_flag = "lose"
        if next_gs.menu_state in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
            self.in_game_frame_counter += 1

        observation = form_state(next_gs, self.ports[0], self.ports[1], stage_mode=1)
        reward = calculate_reward(prev_gs, next_gs, self.ports[0], self.ports[1])
        
        self.prev_gamestate = next_gs
        if button is not None:
            self.ctrls[0].release_button(button)
        self.ctrls[0].tilt_analog(enums.Button.BUTTON_MAIN, 0.5, 0.5)
        self.ctrls[0].flush()
        
        terminated = False
        truncated = False
        
        if self.game_over_flag is not None:
            terminated = True
            if next_gs.menu_state in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
                final_reward = 0.0
                if self.game_over_flag == "win":
                    final_reward = 1.0 
                elif self.game_over_flag == "lose":
                    final_reward = -1.0
                reward += final_reward
        #elif next_gs.menu_state in (Menu.CHARACTER_SELECT, Menu.STAGE_SELECT):
        #    terminated = True
        info = {
            "frame": self.in_game_frame_counter,
            "game_state": str(next_gs.menu_state),
            "bot_stocks": next_gs.players[self.ports[0]].stock,
            "cpu_stocks": next_gs.players[self.ports[1]].stock,
            "bot_percent": next_gs.players[self.ports[0]].percent,
            "cpu_percent": next_gs.players[self.ports[1]].percent,
            "game_over_flag": self.game_over_flag
        }
        self.total_reward = self.total_reward + reward
        if terminated:
            print(f"DEBUG: Game Over - {self.game_over_flag}")
            print(f"DEBUG: Final Reward: {self.total_reward}")
        
        #print(f"frame: {self.in_game_frame_counter}, menu_state: {next_gs.menu_state}")
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass 
    
    def close(self):
        for ctrl in self.ctrls:
            ctrl.disconnect()
        self.console.stop()
