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

    base_height_l2,  # Aggiunta per il reward di altezza del torso
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
            lin_vel_x=(0.0, 0.0),   #starting from zero for stablity (Curriculum Training)
            lin_vel_y=(0.0, 0.0),   
            ang_vel_z=(0.0, 0.0),   
        ),
    )


@configclass
class ActionsCfg:
    """Action: joint position targets for legs + torso."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
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
class RewardStabilityCfg:
    """Reward terms for stable standing."""
    # Smooth actions
    action_rate = RewTerm(func=action_rate_l2, weight=-0.01)

    # Failure penalty
    terminating = RewTerm(func=is_terminated, weight=-2.0)
    
    # Stay alive
    alive = RewTerm(func=is_alive, weight=0.15)

    forward_velocity = RewTerm(
        func=track_lin_vel_xy_exp, 
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25}, # Poiché v_d=(0,0), premia l'immobilità lineare
    )


    # Penalise vertical base velocity (no bouncing)
    lin_vel_z = RewTerm(
        func=lin_vel_z_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # HORIZOTNAL ANG VEL Penalise rolling/pitching of the base, keep the robot up right
    ang_vel_xy = RewTerm(
        func=ang_vel_xy_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # HIP-pose DEVATION , to avoid waist rotation
    ang_vel_z = RewTerm(
        func=mdp.ang_vel_z_l2,
        weight=-1.0,    
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ORIENTATION DEVIATION Keep torso upright, torso inclination 
    flat_orientation = RewTerm(
        func=flat_orientation_l2,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    #BASE HEIGHT PENALTY  tende a mantenere il torso del robot all'altezza target nominale
    base_height = RewTerm(
        func=base_height_l2, # o root_height_l2 a seconda dell'esatta funzione Isaac Lab
        weight=-3.0,            # paper -10 
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.74, # Altezza nominale del torso (es. per Unitree G1)
        },
    )

    # DOF velocity penalty, penalise strong joint acc 
    joint_vel = RewTerm(
        func=joint_vel_l1,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )



    # # Tieni braccia ferme
    # joint_pos_arms = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[".*_shoulder_.*", ".*_elbow.*"]
    #         )
    #     },
    # )

    joint_pos_hip_roll = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.5, 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])
        },
    )

    # Evita che le gambe ruotino verso l'interno incrociandosi
    joint_pos_hip_yaw = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.5, 
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint"]
            )
        },
    )


@configclass
class RewardsCfg(RewardStabilityCfg):
    

    forward_velocity = RewTerm(
        func=mdp.linear_velocity_reward,  # Usa la funzione corretta
        weight=3.0,
        params={"command_name": "base_velocity"},
    )

    # Feet air time - usa versione migliorata
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_reward,  # Usa la tua custom migliorata
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 0.15,  # Più basso per iniziare
            "min_air_time": 0.05,
        },
    )
    
    # Contact alternation - esplicitamente
    feet_alternation = RewTerm(
        func=mdp.feet_contact_alternation,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 0.5,
            "smooth_transition": True,
        },
    )

    #Simmetria della gambe
    leg_symmetry = RewTerm(
        func=mdp.leg_symmetry_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    foot_flat = RewTerm(
        func=mdp.foot_flat_standing_reward,
        weight=1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "target_force": 20.0}
    )
    


    # # Min. torso ang. vel:  termine di regolarità/smoothness del torso
    # torso_ang_vel_reg = RewTerm(
    #     func=mdp.torso_angular_velocity_tracking,
    #     weight=2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # # Waist deviations (Deviazioni del bacino/vita rispetto al target): 1.0 ciascuno
    # waist_pitch = RewTerm(func=mdp.waist_pitch_deviation, weight=1.0)
    # waist_roll = RewTerm(func=mdp.waist_roll_deviation, weight=1.0)
    # waist_yaw = RewTerm(func=mdp.waist_yaw_deviation, weight=1.0)

    # # Torso yaw smoothness: 0.8
    # torso_yaw_smoothness = RewTerm(func=mdp.torso_yaw_smoothness, weight=0.8)

    # # Shoulder roll control: 3.0
    # shoulder_roll = RewTerm(func=mdp.shoulder_roll_control, weight=3.0)

    # # =========================================================================
    # # HUMAN-LIKE ARM SWING (Stile naturale delle braccia - Sempre attivo)
    # # =========================================================================
    
    # # Arm-leg momentum balance (Bilanciamento del momento braccia-gambe): 5.0
    # arm_leg_momentum = RewTerm(func=mdp.arm_leg_momentum_balance, weight=5.0)

    # # Human-like arm swing energy: 0.3
    # arm_swing_energy = RewTerm(func=mdp.arm_swing_energy, weight=0.3)

    # # Elbow phase tracking (Tracciamento della fase del gomito rispetto al passo): 2.5
    # elbow_phase = RewTerm(func=mdp.elbow_phase_tracking, weight=2.5)

    # # Arm swing symmetry: 2.0
    # arm_swing_symmetry = RewTerm(func=mdp.arm_swing_symmetry, weight=2.0)

    # # Arm swing-leg amp. match (Accoppiamento ampiezza braccia-gambe): 1.0
    # arm_leg_amplitude_match = RewTerm(func=mdp.arm_leg_amplitude_match, weight=1.0)


    # # =========================================================================
    # # 2. WALKING-SPECIFIC REWARDS (Attivi solo con Gait ID = Walking)
    # # =========================================================================
    
    # # Feet swing height penalty: -15.0
    # # Penalizza se i piedi si alzano troppo o troppo poco rispetto alla traiettoria ideale
    # feet_swing_height = RewTerm(
    #     func=mdp.feet_swing_height_penalty,
    #     weight=-15.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "gait_id_target": 1} 
    #     # Passiamo 'gait_id_target': la funzione azzererà il reward se l'ambiente non è in modalità Walking
    # )

    # # Contact (Walking contact consistency): 1.0
    # # Premia il contatto alternato corretto dei piedi durante la camminata
    # walking_contact = RewTerm(
    #     func=mdp.walking_contact_pattern,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "gait_id_target": 1}
    # )

    # # Straight knee (Ginocchia tese nella fase di appoggio): 0.1
    # # Questo è il termine cruciale del paper per evitare che l'umanoide cammini "accovacciato"
    # straight_knee = RewTerm(
    #     func=mdp.straight_knee_bonus,
    #     weight=0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "gait_id_target": 1}
    # )

    # # Feet drag penalty (Penalità se trascina i piedi a terra): -0.5
    # feet_drag = RewTerm(
    #     func=mdp.feet_drag_penalty,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "gait_id_target": 1}
    # )
    




@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=bad_orientation,
        params={"limit_angle": math.radians(80)},
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
class G1_gait_ControllerEnvCfg(ManagerBasedRLEnvCfg):
    """Full environment configuration for G1 forward locomotion."""

    scene: G1ControllerSceneCfg = G1ControllerSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()  
    events: EventCfg = EventCfg()
    # rewards: RewardsCfg = RewardsCfg()
    rewards: RewardStabilityCfg = RewardStabilityCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4           # policy a 50 Hz
        self.episode_length_s = 20.0
        self.sim.dt = 0.005           # fisica a 200 Hz
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.0, 3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.8)