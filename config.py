from melee import Character, Stage
#MODE = "Headless" 
MODE = "NoGui"
DOLPHIN_PATH = f"Images/{MODE}/squashfs-root/usr/bin/"  
SLIPPI_ADDR = "127.0.0.1"
ISO_PATH = "ssbm.iso"
SLIPPI_PORT = 51489
PORTS = [1, 2] 
COSTUMES = [1, 2] 

CHARACTER_OPTIONS = {
    0: [Character.KIRBY],
    1: [
        Character.MARIO,
        #Character.PIKACHU,
        #Character.YOSHI,
        #Character.KIRBY,
        #Character.SAMUS,
        #Character.DK,
        #Character.LINK,
        #Character.MARIO
    ]
}


LEVEL_OPTIONS = {
    0: [0],  # agent (0)
    1: [9]   # CPU Always (9)
}

#
STAGE_OPTIONS = [Stage.FINAL_DESTINATION]

CONFIG = {
    "dolphin_path": DOLPHIN_PATH,
    "slippi_address": SLIPPI_ADDR,
    "iso_path": ISO_PATH,
    "slippi_port": SLIPPI_PORT,
    "ports": PORTS,
    "characters": CHARACTER_OPTIONS,
    "levels": LEVEL_OPTIONS,
    "costumes": COSTUMES,
    "stages": STAGE_OPTIONS,
    #"gfx_backend": "Null",
    "disable_audio": True,
    #"use_exi_inputs": True,
    #"enable_ffw": True,
    #"polling_mode": True,
    #"blocking_input": True,
}