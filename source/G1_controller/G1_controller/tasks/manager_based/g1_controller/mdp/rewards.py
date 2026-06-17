# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    target = 0.0
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.mean(torch.square(joint_pos - target), dim=1)

def linear_velocity_reward(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
    """Reward lineare = velocità raggiunta / velocità desiderata."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    current_vel = asset.data.root_lin_vel_b[:, 0]
    desired_vel = command[:, 0]
    
    # Solo velocità positiva (avanti)
    positive_vel = torch.clamp(current_vel, min=0.0)
    
    # Normalizzata rispetto al target (max 1.0)
    reward = torch.clamp(positive_vel / (desired_vel + 0.01), max=1.0)
    
    return reward
   

def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])

# def feet_air_time_reward( env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.4,) -> torch.Tensor:
#     """Reward alternating foot contact pattern. Rewards the robot when feet alternate contact with the ground. Based on air time tracking from ContactSensor."""
    
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
#     # Air time since last contact for each foot [num_envs, num_feet]
#     air_time = contact_sensor.data.last_air_time
    
#     # Contact state [num_envs, num_feet] — True if in contact
#     in_contact = contact_sensor.data.net_forces_w_history[:, 0, :, 2] > 1.0
    
#     # Reward only when foot lands after sufficient air time
#     # Shape: [num_envs, num_feet]
#     reward = torch.clamp(air_time - threshold, min=0.0) * in_contact.float()
    
#     # Sum over feet [num_envs]
#     return reward.sum(dim=-1)

def feet_air_time_reward(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 0.2,       # Soglia target ideale di volo (es. 0.2 secondi)
    min_air_time: float = 0.05,   # Minimo per considerare il piede sollevato
) -> torch.Tensor:
    """Reward foot air time during walking, balancing landing impact and continuous flight."""
    
    # 1. Recupera il sensore tramite il manager della scena
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. Ottieni lo stato di contatto (True se la forza verticale Z > 1.0 Newton)
    # contact_sensor.data.net_forces_w ha shape [num_envs, num_bodies, 3]
    # Selezioniamo tutte le env, tutti i corpi (i 2 piedi) e l'asse Z (indice 2)
    foot_forces = contact_sensor.data.net_forces_w[:, :, 2]
    in_contact = foot_forces > 1.0
    
    # 3. Ottieni il tempo di volo corrente accumulato da ciascun piede
    current_air_time = contact_sensor.data.current_air_time
    
    # REWARD 1: Landing Bonus (Logica classica del paper)
    # Premia quando il piede tocca terra (in_contact) dopo essere rimasto in volo oltre la soglia
    # Usiamo 'last_air_time' che si congela all'istante dell'impatto per premiare il landing
    last_air_time = contact_sensor.data.last_air_time
    landing_bonus = torch.clamp(last_air_time - threshold, min=0.0) * in_contact.float()
    
    # REWARD 2: Continuous Flight Bonus (La tua ottima intuizione)
    # Premia ad ogni step in cui il piede è in aria (~in_contact) per incentivare il movimento di swing
    air_time_bonus = torch.clamp(current_air_time - min_air_time, min=0.0) * (~in_contact).float()
    air_time_bonus = air_time_bonus * 0.5  # Peso ridotto per non destabilizzare lo swing
    
    # REWARD 3: Integrazione Alternanza (Prende la funzione mdp locale)
    # Passiamo la soglia corretta per il sensore
    alternation_bonus = feet_contact_alternation(env, sensor_cfg, threshold=0.5)
    
    # Somma sui piedi (dim=-1) e combina i termini
    total_reward = landing_bonus.sum(dim=-1) + air_time_bonus.sum(dim=-1) + alternation_bonus * 0.3
    
    return total_reward

def feet_drag_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg, 
) -> torch.Tensor:
    """Penalizza il trascinamento dei piedi sul terreno (alta velocità orizzontale durante il contatto)."""
    
    # 1. Recupera l'asset del robot e il sensore di contatto
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. Ottieni la velocità lineare dei corpi associati ai piedi
    # body_names deve coincidere con i link dei piedi configurati in SceneEntityCfg
    # robot.data.body_lin_vel_w ha shape [num_envs, num_bodies, 6] (i primi 3 sono lin_vel, gli altri 3 ang_vel)
    # Isaac Lab espone body_lin_vel_w per le velocità lineari [num_envs, num_bodies, 3]
    
    # Estraiamo gli indici dei corpi dei piedi nella cinematica del robot
    foot_body_ids = asset_cfg.body_ids
    feet_vel_w = robot.data.body_lin_vel_w[:, foot_body_ids, :2] # Prendiamo solo gli assi X e Y (:2)
    
    # Calcoliamo la norma della velocità orizzontale (quanto velocemente slitta il piede)
    feet_speed_xy = torch.norm(feet_vel_w, dim=-1) # shape: [num_envs, num_feet]
    
    # 3. Identifica se il piede è a contatto con il terreno (forza Z > 1.0N)
    foot_forces = contact_sensor.data.net_forces_w[:, :, 2]
    in_contact = (foot_forces > 1.0).float()
    
    # 4. Calcola la penalità: Velocità_XY * In_Contatto
    # Più il piede striscia velocemente mentre è a terra, più la penalità sale
    drag_penalty = (feet_speed_xy ** 2) * in_contact
    
    # Somma il contributo di entrambi i piedi e restituisci il valore negativo (essendo una penalità)
    # Nota: In Isaac Lab restituiamo il valore positivo del calcolo poiché il segno '-' 
    # viene applicato automaticamente dal 'weight=-0.5' dentro la classe RewardsCfg.
    return drag_penalty.sum(dim=-1)




def feet_contact_alternation(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 1.0,  # Ridotto per essere più sensibile
    smooth_transition: bool = True,
) -> torch.Tensor:
    """
    Reward for alternating foot contacts with temporal smoothing.
    
    Versione migliorata con memoria temporale e penalità per doppio appoggio.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces for all feet
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    
    # Detect feet from sensor configuration
    body_names = sensor_cfg.body_names
    if body_names is None:
        body_names = env.scene.sensors[sensor_cfg.name].body_names
    
    # Find left and right feet indices
    left_idx = None
    right_idx = None
    for i, name in enumerate(body_names):
        if "left" in name.lower():
            left_idx = i
        elif "right" in name.lower():
            right_idx = i
    
    if left_idx is None or right_idx is None:
        # Fallback: assume first two are left and right
        left_idx, right_idx = 0, 1
    
    # Binary contact detection
    left_contact = torch.norm(contact_forces[:, left_idx, :], dim=-1) > threshold
    right_contact = torch.norm(contact_forces[:, right_idx, :], dim=-1) > threshold
    
    # REWARD 1: Alternation (XOR) - core reward
    alternation = (left_contact ^ right_contact).float()
    
    # REWARD 2: Penalty for double support (both feet on ground)
    double_support = (left_contact & right_contact).float()
    double_support_penalty = -0.5 * double_support
    
    # REWARD 3: Penalty for no support (both in air - unstable)
    no_support = (~left_contact & ~right_contact).float()
    no_support_penalty = -1.0 * no_support
    
    # REWARD 4: Symmetry bonus (both feet spend similar time in air)
    if smooth_transition:
        # Track contact history (richiede buffer - implementazione semplificata)
        # Per ora usiamo un approccio senza memoria
        left_air_time = contact_sensor.data.last_air_time[:, left_idx]
        right_air_time = contact_sensor.data.last_air_time[:, right_idx]
        
        # Penalize large asymmetry in air time
        air_time_diff = torch.abs(left_air_time - right_air_time)
        symmetry_bonus = torch.exp(-air_time_diff * 5.0)  # Decay rapido
        symmetry_bonus = symmetry_bonus * 0.2  # Piccolo bonus
    else:
        symmetry_bonus = 0.0
    
    # Combine all components
    reward = (
        alternation * 1.0 +           # Alternation reward
        double_support_penalty +       # Discourage double stance
        no_support_penalty +           # Heavily penalize flight
        symmetry_bonus                 # Encourage symmetric walking
    )
    
    return reward

def leg_symmetry_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward symmetric leg motion.
    
    Penalizes asymmetric hip pitch positions between left and right legs.
    Encourages alternating gait pattern.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get hip pitch joint indices for left and right
    left_hip_ids = [
        i for i, name in enumerate(asset.joint_names)
        if "left_hip_pitch" in name
    ]
    right_hip_ids = [
        i for i, name in enumerate(asset.joint_names)
        if "right_hip_pitch" in name
    ]
    
    left_hip = asset.data.joint_pos[:, left_hip_ids]   # [num_envs, 1]
    right_hip = asset.data.joint_pos[:, right_hip_ids]  # [num_envs, 1]
    
    # Symmetric gait: left and right should be opposite phase
    # When left is forward (positive), right should be backward (negative)
    anti_phase = left_hip + right_hip  # should be ~0 for symmetric gait
    
    return -torch.sum(torch.square(anti_phase), dim=-1)

def foot_flat_standing_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_force: float = 20.0,
) -> torch.Tensor:
    """
    Premia quando ENTRAMBI i piedi sono piatti.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    vertical_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    
    # Contatto buono (bool)
    left_good = vertical_forces[:, 0] > target_force
    right_good = vertical_forces[:, 1] > target_force
    
    # Entrambi buoni
    both_flat = (left_good & right_good).float()
    
    # Solo uno buono (penalità leggera)
    one_good = ((left_good ^ right_good)).float() * -0.3
    
    # Nessuno buono (penalità maggiore)
    none_good = ((~left_good & ~right_good)).float() * -0.5
    
    return both_flat + one_good + none_good