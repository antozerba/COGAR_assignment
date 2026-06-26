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



from . import mdp


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
    # is_terminated,  <-- Nota: In Isaac Lab si usa spesso 'is_terminated_thin' o penalità dirette
    lin_vel_z_l2,
    ang_vel_xy_l2,
    flat_orientation_l2,
    joint_vel_l1,
    joint_torques_l2,         
    action_rate_l2,
    track_lin_vel_xy_exp,
    # terminations
    time_out,
    bad_orientation,
    root_height_below_minimum,

    
)
from isaaclab.envs.mdp import generated_commands



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

    robot: ArticulationCfg = G1_CFG.replace(
      prim_path="{ENV_REGEX_NS}/Robot",
      spawn=G1_CFG.spawn.replace(
          articulation_props=sim_utils.ArticulationRootPropertiesCfg(
              enabled_self_collisions=True,  # abilita collisioni tra parti del robot
              solver_position_iteration_count=16,
              solver_velocity_iteration_count=8,
          ),
      )
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        update_period= 0.0,
        history_length= 3,
        track_air_time= True

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
            lin_vel_x=(-1.0, 1.0),   
            lin_vel_y=(-0.5, 0.5),   
            ang_vel_z=(0.0, 0.0),   
        ),
    )

@configclass
class CommandsCfgPlay:
    """Command: fixed forward velocity."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),   
            lin_vel_y=(0.0, 0.0),   
            ang_vel_z=(0.0, 0.0),   
        ),
    )


@configclass
class ActionsCfg:
    """Action: joint position targets for ALL 29 joints."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        # Corretto: includiamo TUTTI i giunti per mappare il modello 29DOF di MuJoCo
        joint_names=[".*"], 
        scale = 0.25, # Ridotto a 0.25 per evitare movimenti troppo bruschi (soprattutto con il modello 29DOF)
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observations for the locomotion policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations seen by the policy."""

        #Velocity command 3
        velocity_command = ObsTerm(
            func= generated_commands,
            params={"command_name": "base_velocity"}
        )

        # Base state  3 +3 + 3
        base_lin_vel = ObsTerm(func=base_lin_vel)
        base_ang_vel = ObsTerm(func=base_ang_vel)
        projected_gravity = ObsTerm(func=projected_gravity)

        # Joint state 29 + 29
        joint_pos_rel = ObsTerm(func=joint_pos_rel)
        joint_vel_rel = ObsTerm(func=joint_vel_rel)

        # Last action (for smoothness reward) 29
        last_action = ObsTerm(func=last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    #Total size = 99
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Randomization events."""
    #Introduce some noise in the position and give a little volocity so that the robot need to adapt
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
    #Resetting robot to its initial position but with a little bit of error

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

    # ========================================
    # TRACKING REWARDS
    # ========================================
    
    # Track linear velocity (xy) - exponential kernel
    tracking_lin_vel = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    
    # Track angular velocity (yaw) - exponential kernel
    tracking_ang_vel = RewTerm(
        func=mdp.tracking_ang_vel_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    alive = RewTerm(func=is_alive, weight=0.3)

    # ========================================
    # BASE REWARDS 
    # ========================================
    
    # Penalize angular velocity in xy (roll/pitch rates)
    ang_vel_xy = RewTerm(
        func=ang_vel_xy_l2,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # # # Penalize orientation deviation (torso specific target [0.073, 0, 1])
    # # orientation = RewTerm(
    # #     func=mdp.orientation_torso_l2,
    # #     weight=-2.0,
    # #     params={"asset_cfg": SceneEntityCfg("robot")},
    # # ) 
    

    # ========================================
    # FEET REWARDS
    # ========================================
    
    # Reward for proper feet air time (0.2s - 0.5s range)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 0.1,  # Considera il piede in volo se la forza scende sotto i 0.1N
        },
    )
    
    # Penalize feet slip (body vel when in contact)
    feet_slip = RewTerm(
        func=mdp.feet_slip_penalty,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


    # ========================================
    # OTHER REWARDS 
    # ========================================
    
    # Penalize joint deviation when command is near zero
    stand_still = RewTerm(
        func=mdp.stand_still_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Heavy penalty for termination
    termination = RewTerm(
        func=mdp.termination_penalty,
        weight=-2.0,
    )
    #TODO: put sensor
    # # Penalize hand-thigh collisions (placeholder)
    # collision = RewTerm(
    #     func=mdp.collision_hands_thighs_penalty,
    #     weight=-0.1,
    # )
    
    # Penalize excessive contact forces (>500N)
    contact_force = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "max_force": 500.0,
        },
    )

    # ========================================
    # POSE REWARDS 
    # ========================================
    
    # Penalize knee deviation from default (L1)
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_knee_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )
    
    # Penalize hip deviation from default (L1, with lateral vel handling)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_hip_l1,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]),
        },
    )
    
    # Penalize joint positions near soft limits
    dof_pos_limits = RewTerm(
        func=mdp.dof_pos_limits_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "soft_factor": 0.95,
        },
    )
    
    # Penalize pose deviation from default (L2, ALL joints)
    pose = RewTerm(
        func=mdp.pose_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=bad_orientation,
        params={"limit_angle": math.radians(70)},
    )



##
# Environment configuration
##

@configclass
class G1_29dof_ControllerEnvCfg(ManagerBasedRLEnvCfg):
    """Full environment configuration for G1 forward locomotion."""

    scene: G1ControllerSceneCfg = G1ControllerSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfgPlay = CommandsCfgPlay()  
    # commands: CommandsCfg = CommandsCfg()  
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