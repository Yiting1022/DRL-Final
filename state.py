import numpy as np
import melee
from melee import Menu

def form_state(gs, player_port, opp_port, stage_mode=1, normalize=True):
    player = gs.players[player_port]
    opp = gs.players[opp_port]

    def closest_projectile_position(player, projectiles):
        try:
            closest_proj = None
            min_distance = float('inf')
            for proj in projectiles:
                if proj.owner == opp_port: 
                    distance = ((proj.position.x - player.position.x) ** 2 + (proj.position.y - player.position.y) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_proj = proj
            if closest_proj:
                return closest_proj.position.x, closest_proj.position.y
            else:
                return 0.0, 0.0  
        except Exception:
            return 0.0, 0.0 

    def ply_feats(p):
        proj_x, proj_y = closest_projectile_position(p, gs.projectiles)
        return [
            stage_mode,
            p.percent,
            p.position.x,
            p.position.y,
            p.speed_ground_x_self,
            p.speed_y_self,
            float(p.on_ground),
            float(p.off_stage),
            p.jumps_left,
            float(p.facing),
            float(p.invulnerable),
            p.action.value,
            p.action_frame,
            p.hitstun_frames_left,
            p.hitlag_left,
            p.stock,
            p.shield_strength,
            p.speed_x_attack,
            p.speed_y_attack,
            p.speed_air_x_self,
            p.invulnerability_left,
            proj_x,
            proj_y
        ]
        
        

    player_feats = ply_feats(player)
    opp_feats = ply_feats(opp)

    state = np.array(player_feats + opp_feats, dtype=np.float32)
    
    if normalize:
    
        xy_scale = 0.1
        speed_scale = 0.5
        percent_scale = 0.01
        shield_scale = 0.01
        frame_scale = 0.1

        # 玩家特徵規範化 (索引對應上面的ply_feats順序)
        state[1] *= percent_scale   # percent
        state[2] *= xy_scale        # position.x
        state[3] *= xy_scale        # position.y
        state[4] *= speed_scale     # speed_ground_x_self
        state[5] *= speed_scale     # speed_y_self
        # 7-10是布爾值，不需要縮放
        # 11是action_value
        state[12] *= frame_scale    # action_frame
        # 13是布爾值
        state[14] *= frame_scale    # hitlag_left
        # 15是stock數量
        state[16] *= shield_scale   # shield_strength
        state[17] *= speed_scale    # speed_x_attack
        state[18] *= speed_scale    # speed_y_attack
        state[19] *= speed_scale    # speed_air_x_self
        state[20] *= frame_scale    # invulnerability_left
        state[21] *= xy_scale       # proj_x
        state[22] *= xy_scale       # proj_y
        
        # 對手特徵規範化 (索引+23)
        state[24] *= percent_scale  # percent
        state[25] *= xy_scale       # position.x
        state[26] *= xy_scale       # position.y
        state[27] *= speed_scale    # speed_ground_x_self
        state[28] *= speed_scale    # speed_y_self
        # 29-32是布爾值
        # 34是action_value
        state[35] *= frame_scale    # action_frame
        # 36是布爾值
        state[37] *= frame_scale    # hitlag_left
        # 38是stock數量
        state[39] *= shield_scale   # shield_strength
        state[40] *= speed_scale    # speed_x_attack
        state[41] *= speed_scale    # speed_y_attack
        state[42] *= speed_scale    # speed_air_x_self
        state[43] *= frame_scale    # invulnerability_left
        state[44] *= xy_scale       # proj_x
        state[45] *= xy_scale       # proj_y
    return state

def is_dying(player):
    return player.action.value <= 0xA

def process_deaths(prev_gs, next_gs, port):
    if prev_gs is None or next_gs is None:
        return 0
    prev_dead = is_dying(prev_gs.players[port])
    next_dead = is_dying(next_gs.players[port])
    death_percent = prev_gs.players[port].percent
    return ((not prev_dead) and next_dead), death_percent

def process_damages(prev_gs, next_gs, port):
    if prev_gs is None or next_gs is None:
        return 0
    prev_percent = prev_gs.players[port].percent
    next_percent = next_gs.players[port].percent
    
    return max(next_percent - prev_percent, 0)

def get_distance(gs, player_port, opp_port):

    x0 = gs.players[player_port].position.x
    y0 = gs.players[player_port].position.y
    x1 = gs.players[opp_port].position.x
    y1 = gs.players[opp_port].position.y
    
    dx = x1 - x0
    dy = y1 - y0
    
    return -np.sqrt(dx*dx + dy*dy)

def offstage_distance(gs, player_port):
    player_y = gs.players[player_port].position.y
    if player_y < 0:  # Offstage
        return -player_y
    return 0

def calculate_death_loss(percent, death, percent_ratio=0.005):
    if death == 0:
        return 0.0
    return 1 - (percent * percent_ratio) if 1 - (percent * percent_ratio) > 0 else 0.0


def calculate_reward(prev_gs, next_gs, player_port, opp_port, damage_ratio=0.005, gamma=0.99):
    if prev_gs is None or next_gs is None:
        return 0.0

    player_death, player_death_percent = process_deaths(prev_gs, next_gs, player_port)
    opp_death, opp_death_percent = process_deaths(prev_gs, next_gs, opp_port)
    player_damage = process_damages(prev_gs, next_gs, player_port)
    opp_damage = process_damages(prev_gs, next_gs, opp_port)
    #player_distance = get_distance(next_gs, player_port, opp_port)
    #player_offstage_distance = offstage_distance(next_gs, player_port)
    #opp_distance = get_distance(next_gs, opp_port, player_port)
    #opp_offstage_distance = offstage_distance(next_gs, opp_port)
    
    
    player_loss = calculate_death_loss(player_death_percent, player_death) + damage_ratio * player_damage
    opp_loss = calculate_death_loss(opp_death_percent, opp_death) + damage_ratio * opp_damage

    base_reward = opp_loss - player_loss
    
    #reward shaping
    
    # not offstage bonus
    # shape_reward = 0.0
    #if next_gs.players[player_port].off_stage:
    #    shape_reward -=  0.01 * (abs(prev_gs.players[player_port].position.x) - abs(next_gs.players[player_port].position.x))
    
    # close combat bonus
    #shape_reward += 0.01 * get_distance(next_gs, player_port, opp_port)

    return base_reward
