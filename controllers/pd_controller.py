import numpy as np
import sys
import os

# Aggiunge la cartella root del progetto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.g1_env import G1Env


# ----------------------------------------------------------------------
# Posa di stand del G1 29 DOF
# Ordine giunti: left leg, right leg, waist, left arm, right arm
# ----------------------------------------------------------------------
STAND_POS = np.array([
    # Left leg (6 giunti)
    -0.1,   # left_hip_pitch_joint
     0.0,   # left_hip_roll_joint
     0.0,   # left_hip_yaw_joint
     0.3,   # left_knee_joint
    -0.2,   # left_ankle_pitch_joint
     0.0,   # left_ankle_roll_joint
    # Right leg (6 giunti)
    -0.1,   # right_hip_pitch_joint
     0.0,   # right_hip_roll_joint
     0.0,   # right_hip_yaw_joint
     0.3,   # right_knee_joint
    -0.2,   # right_ankle_pitch_joint
     0.0,   # right_ankle_roll_joint
    # Waist (3 giunti)
     0.0,   # waist_yaw_joint
     0.0,   # waist_roll_joint
     0.0,   # waist_pitch_joint
    # Left arm (7 giunti)
     0.0,   # left_shoulder_pitch_joint
     0.0,   # left_shoulder_roll_joint
     0.0,   # left_shoulder_yaw_joint
     0.0,   # left_elbow_joint
     0.0,   # left_wrist_roll_joint
     0.0,   # left_wrist_pitch_joint
     0.0,   # left_wrist_yaw_joint
    # Right arm (7 giunti)
     0.0,   # right_shoulder_pitch_joint
     0.0,   # right_shoulder_roll_joint
     0.0,   # right_shoulder_yaw_joint
     0.0,   # right_elbow_joint
     0.0,   # right_wrist_roll_joint
     0.0,   # right_wrist_pitch_joint
     0.0,   # right_wrist_yaw_joint
], dtype=np.float64)


class PDStandController:
    """
    Controllore PD che mantiene il robot G1 in posizione di stand.

    Per ogni giunto calcola la coppia da applicare come:
        tau = Kp * (q_target - q_current) - Kd * dq_current

    dove:
        q_target   = posizione angolare desiderata (STAND_POS)
        q_current  = posizione angolare corrente dal sensore
        dq_current = velocità angolare corrente dal sensore
        Kp         = guadagno proporzionale
        Kd         = guadagno derivativo (smorzamento)
    """

    def __init__(self):
        # --- Guadagni PD ---
        # Le gambe richiedono guadagni più alti perché devono sostenere il peso
        # Le braccia e il busto richiedono guadagni più bassi
        # Formato: array di dimensione 29, uno per giunto

        kp_legs  = 200.0   # proporzionale gambe
        kd_legs  =   5.0   # derivativo gambe

        kp_waist = 150.0   # proporzionale busto
        kd_waist =   3.0   # derivativo busto

        kp_arms  =  40.0   # proporzionale braccia
        kd_arms  =   2.0   # derivativo braccia

        # Costruisce i vettori Kp e Kd per tutti i 29 giunti
        self.Kp = np.array([
            # Left leg (6)
            kp_legs, kp_legs, kp_legs, kp_legs, kp_legs, kp_legs,
            # Right leg (6)
            kp_legs, kp_legs, kp_legs, kp_legs, kp_legs, kp_legs,
            # Waist (3)
            kp_waist, kp_waist, kp_waist,
            # Left arm (7)
            kp_arms, kp_arms, kp_arms, kp_arms, kp_arms, kp_arms, kp_arms,
            # Right arm (7)
            kp_arms, kp_arms, kp_arms, kp_arms, kp_arms, kp_arms, kp_arms,
        ], dtype=np.float64)

        self.Kd = np.array([
            # Left leg (6)
            kd_legs, kd_legs, kd_legs, kd_legs, kd_legs, kd_legs,
            # Right leg (6)
            kd_legs, kd_legs, kd_legs, kd_legs, kd_legs, kd_legs,
            # Waist (3)
            kd_waist, kd_waist, kd_waist,
            # Left arm (7)
            kd_arms, kd_arms, kd_arms, kd_arms, kd_arms, kd_arms, kd_arms,
            # Right arm (7)
            kd_arms, kd_arms, kd_arms, kd_arms, kd_arms, kd_arms, kd_arms,
        ], dtype=np.float64)

        # Posizione target
        self.q_target = STAND_POS.copy() #il nostro target è tenere il robot nella posizione iniziale

    def compute_action(self, obs: dict) -> np.ndarray:
        """
        Calcola le coppie da applicare ai 29 giunti.

        Parametri
        ----------
        obs : dizionario restituito da G1Env.get_obs()

        Restituisce
        ----------
        tau : array 29 con le coppie in Nm
        """
        q_current  = obs["joint_pos"]   # posizioni correnti [rad]
        dq_current = obs["joint_vel"]   # velocità correnti [rad/s]

        # Legge PD: tau = Kp * (q_des - q) - Kd * dq
        tau = self.Kp * (self.q_target - q_current) - self.Kd * dq_current #PD per tenere il robot in STAND_POSE

        return tau


# ----------------------------------------------------------------------
# Test: avvia la simulazione con il controllore PD di stand
# ----------------------------------------------------------------------
if __name__ == "__main__":

    xml_path = os.path.expanduser(
        "~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    )

    print("Avvio simulazione con controllore PD di stand...")
    env        = G1Env(xml_path=xml_path, render=True)
    controller = PDStandController()

    obs = env.reset()

    print("Simulazione in corso - chiudi la finestra per fermare")
    print(f"{'Step':>6}  {'Altezza':>8}  {'Roll':>8}  {'Pitch':>8}  {'Caduto':>8}")
    print("-" * 50)

    step = 0
    while env.viewer is not None and env.viewer.is_running():

        # Calcola azione dal controllore
        action = controller.compute_action(obs)

        # Step simulazione
        obs, done, info = env.step(action)

        # Stampa info ogni 200 step (ogni secondo di simulazione)
        if step % 200 == 0:
            h     = obs["base_pos"][2]
            roll  = np.rad2deg(obs["base_euler"][0])
            pitch = np.rad2deg(obs["base_euler"][1])
            print(f"{step:>6}  {h:>8.3f}m  {roll:>7.2f}°  {pitch:>7.2f}°  {'SI' if done else 'no':>8}")

        step += 1

    print("\nSimulazione terminata.")