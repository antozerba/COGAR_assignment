"""Custom MDP functions for G1 joystick task (ported from MuJoCo)."""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tracking_ang_vel_exp(
    env: "ManagerBasedRLEnv", 
    command_name: str, 
    std: float = 0.25
) -> torch.Tensor:
    """Reward for tracking angular velocity command (yaw only) using exponential kernel.
    
    From MuJoCo: exp(-square(cmd[2] - ang_vel[2]) / sigma)
    """
    commands = env.command_manager.get_command(command_name)
    ang_vel = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error = torch.square(commands[:, 2] - ang_vel)
    return torch.exp(-error / std)


def orientation_torso_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation of torso gravity vector from target [0.073, 0, 1].
    
    This is specific to G1 where the torso has a slight forward lean in the default pose.
    From MuJoCo: sum(square(gravity - [0.073, 0, 1]))
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    # Target is unnormalized in MuJoCo code (specific to G1 model)
    target = torch.tensor([0.073, 0.0, 1.0], device=env.device)
    return torch.sum(torch.square(gravity - target), dim=1)


def feet_slip_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalize body velocity when feet are in contact.
    
    From MuJoCo: sum(||body_vel[:2]|| * contact)
    This discourages the body from moving while feet are planted.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.root_lin_vel_w[:, :2]
    body_vel_norm = torch.norm(body_vel, dim=1)
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w
    # Contact is True if any foot has significant force
    contact = torch.any(torch.norm(contact_forces[:, :, :2], dim=-1) > 1.0, dim=1).float()
    
    return body_vel_norm * contact


def stand_still_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default when command is near zero.
    
    From MuJoCo: sum(|qpos - default|) * (cmd_norm < 0.01)
    Only active when the robot should be standing still.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands, dim=1)
    
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)
    
    mask = (cmd_norm < 0.01).float()
    return deviation * mask


def termination_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Heavy penalty for termination.
    
    From MuJoCo: done (binary, weighted at -100)
    """

    done = env.termination_manager.compute()
    return done.float()
    


def contact_force_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    max_force: float = 500.0,
) -> torch.Tensor:
    """Penalize contact forces exceeding threshold.
    
    From MuJoCo: sum(clip(|force_z| - max_force, min=0))
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w
    normal_forces = torch.abs(contact_forces[:, :, 2])
    excess = torch.clamp(normal_forces - max_force, min=0.0)
    return torch.sum(excess, dim=1)


def joint_deviation_hip_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", 
        joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]
    ),
) -> torch.Tensor:
    """Penalize hip joint deviation from default with special lateral velocity handling.
    
    From MuJoCo: Allows hip_roll deviation when lateral velocity > 0.1
    Weight is [roll, yaw, roll, yaw] = [1-lateral_mask, 1, 1-lateral_mask, 1]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)
    
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.abs(joint_pos - default_pos)
    
    # Create weights: 0 for hip_roll when lateral vel > 0.1
    lateral_vel_high = (commands[:, 1] > 0.1).float()
    weights = torch.ones_like(error)
    # Assuming order: left_hip_roll, left_hip_yaw, right_hip_roll, right_hip_yaw
    if weights.shape[1] >= 2:
        weights[:, 0] = 1.0 - lateral_vel_high  # left hip_roll
    if weights.shape[1] >= 4:
        weights[:, 2] = 1.0 - lateral_vel_high  # right hip_roll
    
    return torch.sum(error * weights, dim=1)


def joint_deviation_knee_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
) -> torch.Tensor:
    """Penalize knee joint deviation from default. """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.abs(joint_pos - default_pos)
    return torch.sum(error, dim=1)


def dof_pos_limits_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_factor: float = 0.95,
) -> torch.Tensor:
    """Penalize joint positions exceeding soft limits.
    
    From MuJoCo: soft_limits = center ± 0.5 * range * soft_factor
    Penalty = sum of violations
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    lower = asset.data.joint_limits[:, asset_cfg.joint_ids, 0]
    upper = asset.data.joint_limits[:, asset_cfg.joint_ids, 1]
    
    center = (lower + upper) / 2
    range_ = upper - lower
    soft_lower = center - 0.5 * range_ * soft_factor
    soft_upper = center + 0.5 * range_ * soft_factor
    
    below_lower = torch.clamp(soft_lower - joint_pos, min=0.0)
    above_upper = torch.clamp(joint_pos - soft_upper, min=0.0)
    
    return torch.sum(below_lower + above_upper, dim=1)


def pose_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from default pose (L2)."""
    
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - default_pos), dim=1)

