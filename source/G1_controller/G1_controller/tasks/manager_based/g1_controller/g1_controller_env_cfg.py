# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 humanoid locomotion task (walk forward)."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg  # noqa: F401
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_CFG

from isaaclab.envs.mdp import (
    # actions
    JointPositionActionCfg,
    # observations
    base_lin_vel,
    base_ang_vel,
    projected_gravity,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    # commands
    UniformVelocityCommandCfg,
    # events
    reset_root_state_uniform,
    reset_joints_by_offset,
    # rewards
    is_alive,
    is_terminated,
    lin_vel_z_l2,
    ang_vel_xy_l2,
    flat_orientation_l2,
    joint_vel_l1,
    action_rate_l2,
    track_lin_vel_xy_exp,
    # terminations
    time_out,
    bad_orientation,
    root_height_below_minimum,
)

from . import mdp


##
# Scene definition
##


@configclass
class G1ControllerSceneCfg(InteractiveSceneCfg):
    """Scene with G1 humanoid robot on flat ground."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )

    # Distant light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # G1 robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensors on feet
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=True,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command: fixed forward velocity."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),   # sempre 0.5 m/s in avanti
            lin_vel_y=(0.0, 0.0),   # no movimento laterale
            ang_vel_z=(0.0, 0.0),   # no rotazione
        ),
    )


@configclass
class ActionsCfg:
    """Action: joint position targets for legs + torso."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "torso_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observations for the locomotion policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations seen by the policy."""

        # Base state
        base_lin_vel = ObsTerm(func=base_lin_vel)
        base_ang_vel = ObsTerm(func=base_ang_vel)
        projected_gravity = ObsTerm(func=projected_gravity)

        # Joint state
        joint_pos_rel = ObsTerm(func=joint_pos_rel)
        joint_vel_rel = ObsTerm(func=joint_vel_rel)

        # Last action (for smoothness reward)
        last_action = ObsTerm(func=last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Randomization events."""

    reset_robot_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for walking forward."""

    # Primary: walk forward at target velocity
    forward_velocity = RewTerm(
        func=track_lin_vel_xy_exp, 
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25}, 
        #prende la v_d dal comando per fare V_reale - V_desired e setta la standard deviation per exponential in track_lin_vel_xy_exp
    )

    # Stay alive
    alive = RewTerm(func=is_alive, weight=0.5)

    # Penalise vertical base velocity (no bouncing)
    lin_vel_z = RewTerm(
        func=lin_vel_z_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalise rolling/pitching of the base
    ang_vel_xy = RewTerm(
        func=ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Keep torso upright
    flat_orientation = RewTerm(
        func=flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Energy efficiency
    joint_vel = RewTerm(
        func=joint_vel_l1,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

  

    # Smooth actions
    action_rate = RewTerm(func=action_rate_l2, weight=-0.01)

    # Failure penalty
    terminating = RewTerm(func=is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=bad_orientation,
        params={"limit_angle": math.radians(60)},
    )

    base_height = DoneTerm(
        func=root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "minimum_height": 0.3,
        },
    )


##
# Environment configuration
##


@configclass
class G1ControllerEnvCfg(ManagerBasedRLEnvCfg):
    """Full environment configuration for G1 forward locomotion."""

    scene: G1ControllerSceneCfg = G1ControllerSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()  
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4           # policy a 50 Hz
        self.episode_length_s = 20.0
        self.sim.dt = 0.005           # fisica a 200 Hz
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.0, 3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.8)