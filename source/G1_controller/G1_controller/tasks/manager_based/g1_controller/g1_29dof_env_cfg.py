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
            lin_vel_x=(0.4, 0.4),   # sempre 0.5 m/s in avanti
            lin_vel_y=(0.0, 0.0),   # no movimento laterale
            ang_vel_z=(0.0, 0.0),   # no rotazione
        ),
    )


@configclass
class ActionsCfg:
    """Action: joint position targets for ALL 29 joints."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"], 
        scale = 0.25, 
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
    """Reward terms for walking forward."""

    #------------------------

    # Primary: walk forward at target velocity
    forward_velocity = RewTerm(
        func=track_lin_vel_xy_exp, 
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.25}, 
    )

    # Stay alive
    alive = RewTerm(func=is_alive, weight=0.3) # not too high i dont want the robot to start walking strangly just to be alive

    #------------------------

    #CORE EQUILIBRIUM
    # Penalise vertical base velocity (no bouncing)
    lin_vel_z = RewTerm(
        func=lin_vel_z_l2,
        weight=-1.5,
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
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Avoid body turning and force walk straight (tried after 1000 but makes the robot moves only one leg forward )
    ang_vel_yaw = RewTerm(
        func=mdp.ang_vel_z_l2, 
        weight=-1.0,  # Aumentato da -0.5 a -2.0
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    #------------------------

    #EFFIECINCY TERMS
    # Energy efficiency: Velocità dei giunti
    joint_vel = RewTerm(
        func=joint_vel_l1,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Cruciale per MuJoCo: Penalizza coppie elevate (evita tremolii distruttivi)
    joint_torques = RewTerm(
        func=joint_torques_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    #------------------------
    # WALKING STYLE

    # Forza le gambe a rimanere vicine alla postura eretta di default
    joint_pos_hip_roll = RewTerm(
        func=mdp.joint_pos_target_l2,
    
        weight=-0.1, 
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*_hip_roll_joint"] # Corretto con il nome esatto standard del G1
            )
        },
    )

    # Regolarizzazione per evitare che le braccia sfarfallino selvaggiamente
    joint_pos_arms_deviation = RewTerm(
        func=mdp.joint_pos_target_l2, 
    
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*", 
                    ".*_elbow_pitch_joint", 
                    ".*_elbow_roll_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                    ".*_three_joint",
                    ".*_four_joint",
                    ".*_five_joint",
                    ".*_six_joint"
                ]
            )
        },
    )
    #Evitare ancora farfallio
    joint_vel_arms_penalty = RewTerm(
        func=mdp.joint_vel_l1, # Oppure joint_vel_l2 (al quadrato è più fluido)
        weight=-0.01,         # Inizia basso, alza se sono ancora troppo agitate
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint"
                ]
            )
        },
    )
    # Regalizzazione per mantenere il busto (waist) vicino alla posizione eretta, ma con peso molto basso per non interferire troppo con i movimenti naturali del G1
    joint_pos_waist = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["torso_joint"]
            )
        },
    )


    #------------------------
    # CONTACT FEET AIR TIME
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 0.1,  # Considera il piede in volo se la forza scende sotto i 0.1N
        },
    )

    feet_drag = RewTerm(
        func=mdp.feet_drag_penalty,
        weight=-0.25,
        params={
        # Diciamo alla funzione quali body tracciare (devono essere i piedi)
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), 
            # Passiamo il sensore di contatto associato
            "sensor_cfg": SceneEntityCfg("contact_forces")
        },
    )

    # Smooth actions
    action_rate = RewTerm(func=action_rate_l2, weight=-0.05)


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=bad_orientation,
        params={"limit_angle": math.radians(70)},
    )

    base_height = DoneTerm(
        func=root_height_below_minimum,
        params={
            # Corretto: 'pelvis' è il root body effettivo del G1
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"), 
            "minimum_height": 0.5, #the standing position is 0.74 i dont want too low
        },
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