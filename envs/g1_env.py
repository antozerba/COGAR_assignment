import mujoco
import mujoco.viewer
import numpy as np
import time

class G1Env:
    """
    Ambiente MuJoCo per il robot Unitree G1 29 DOF.
    Interfaccia diretta con le API MuJoCo senza SDK o DDS.
    """

    # Timestep della simulazione (coerente con config.py di unitree_mujoco)
    SIMULATE_DT = 0.005  # 200 Hz
    VIEWER_DT   = 0.02   # 50 fps

    def __init__(self, xml_path: str, render: bool = True):
        """
        Parametri
        ----------
        xml_path : percorso al file XML della scena, es. assets/g1/scene_29dof.xml
        render   : se True apre il viewer MuJoCo in modalità passiva
        """
        # --- Caricamento modello ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # Imposta il timestep
        self.model.opt.timestep = self.SIMULATE_DT

        # --- Info modello ---
        self.num_joints  = self.model.nu          # numero attuatori (29 per G1)
        self.num_qpos    = self.model.nq          # dimensione vettore posizione generalizzata
        self.num_qvel    = self.model.nv          # dimensione vettore velocità generalizzata

        # ID del corpo base (torso) - usato per leggere posa e velocità del corpo rigido
        self.torso_id = self.model.body("torso_link").id

        # --- Viewer ---
        self.render = render
        self.viewer = None
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # --- Stato interno ---
        self._step_count  = 0
        self._last_render = time.perf_counter()

        # Salva la configurazione iniziale per il reset
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self, qpos: np.ndarray = None, qvel: np.ndarray = None):
        """
        Riporta la simulazione allo stato iniziale.
        Se qpos/qvel vengono passati, usa quelli; altrimenti usa quelli di default del modello.

        Restituisce l'osservazione iniziale.
        """
        mujoco.mj_resetData(self.model, self.data)

        if qpos is not None:
            # qpos ha dimensione nq: 7 (floating base: 3 pos + 4 quaternion) + 29 giunti = 36
            assert len(qpos) == self.num_qpos, \
                f"qpos atteso di dimensione {self.num_qpos}, ricevuto {len(qpos)}"
            self.data.qpos[:] = qpos

        if qvel is not None:
            assert len(qvel) == self.num_qvel
            self.data.qvel[:] = qvel

        # Propaga le posizioni aggiornate nella cinematica forward
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self.get_obs()

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        Avanza la simulazione di un timestep applicando l'azione.

        Parametri
        ----------
        action : array di dimensione num_joints (29)
                 Rappresenta le coppie di torque desiderate [Nm]
                 oppure le posizioni target se il controllore usa PD.

        Restituisce
        ----------
        obs     : dict con l'osservazione corrente
        done    : bool, True se il robot è caduto
        info    : dict con info aggiuntive (es. step count)
        """
        assert len(action) == self.num_joints, \
            f"action attesa di dimensione {self.num_joints}, ricevuta {len(action)}"

        # Applica l'azione agli attuatori
        self.data.ctrl[:] = action

        # Esegue uno step fisico MuJoCo
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # Aggiorna il viewer se attivo
        if self.render and self.viewer is not None and self.viewer.is_running():
            now = time.perf_counter()
            if now - self._last_render >= self.VIEWER_DT:
                self.viewer.sync()
                self._last_render = now

        obs  = self.get_obs()
        done = self._is_fallen()
        info = {"step": self._step_count, "time": self.data.time}

        return obs, done, info

    # ------------------------------------------------------------------
    # OSSERVAZIONE
    # ------------------------------------------------------------------
    def get_obs(self) -> dict:
        """
        Restituisce lo stato completo del robot come dizionario.

        Campi
        -----
        base_pos        : [x, y, z] posizione del torso nel mondo
        base_quat       : [w, x, y, z] orientamento quaternione del torso
        base_euler      : [roll, pitch, yaw] in radianti
        base_lin_vel    : [vx, vy, vz] velocità lineare del torso
        base_ang_vel    : [wx, wy, wz] velocità angolare del torso
        joint_pos       : array 29 posizioni angolari dei giunti [rad]
        joint_vel       : array 29 velocità angolari dei giunti [rad/s]
        joint_torque    : array 29 coppie applicate [Nm]
        contact_forces  : [FL, FR] forze di contatto verticali piede sinistro/destro [N]
        """
        # --- Posa base ---
        # qpos[0:3]  = posizione xyz del floating base
        # qpos[3:7]  = quaternione wxyz del floating base
        base_pos  = self.data.qpos[0:3].copy()
        base_quat = self.data.qpos[3:7].copy()  # [w, x, y, z] in MuJoCo

        # Converti quaternione in angoli di Eulero (roll, pitch, yaw)
        base_euler = self._quat_to_euler(base_quat)

        # --- Velocità base ---
        # qvel[0:3]  = velocità lineare
        # qvel[3:6]  = velocità angolare
        base_lin_vel = self.data.qvel[0:3].copy()
        base_ang_vel = self.data.qvel[3:6].copy()

        # --- Stato giunti ---
        # qpos[7:]  = posizioni angolari dei 29 giunti
        # qvel[6:]  = velocità angolari dei 29 giunti
        joint_pos    = self.data.qpos[7:].copy()
        joint_vel    = self.data.qvel[6:].copy()
        joint_torque = self.data.actuator_force.copy()

        # --- Contatti piedi ---
        contact_forces = self._get_foot_contact_forces()

        return {
            "base_pos":       base_pos,
            "base_quat":      base_quat,
            "base_euler":     base_euler,
            "base_lin_vel":   base_lin_vel,
            "base_ang_vel":   base_ang_vel,
            "joint_pos":      joint_pos,
            "joint_vel":      joint_vel,
            "joint_torque":   joint_torque,
            "contact_forces": contact_forces,
        }

    # ------------------------------------------------------------------
    # HELPER: rilevamento caduta
    # ------------------------------------------------------------------
    def _is_fallen(self) -> bool:
        """
        Il robot è considerato caduto se:
        - l'altezza del torso scende sotto 0.3 m (da terra)
        - oppure roll o pitch superano 60 gradi
        """
        height = self.data.qpos[2]
        euler  = self._quat_to_euler(self.data.qpos[3:7])
        roll   = abs(euler[0])
        pitch  = abs(euler[1])

        fallen = (height < 0.3) or (roll > np.deg2rad(60)) or (pitch > np.deg2rad(60))
        return bool(fallen)

    # ------------------------------------------------------------------
    # HELPER: forze di contatto ai piedi
    # ------------------------------------------------------------------
    def _get_foot_contact_forces(self) -> np.ndarray:
        """
        Legge la forza di contatto verticale (asse Z) per i due piedi.
        Restituisce [F_left, F_right] in Newton.
        """
        forces = np.zeros(2)

        # Recupera gli ID dei corpi dei piedi dal modello
        try:
            left_foot_id  = self.model.body("left_ankle_roll_link").id
            right_foot_id = self.model.body("right_ankle_roll_link").id
        except Exception:
            return forces  # se i nomi non corrispondono restituisce zeri

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Recupera gli ID dei geom coinvolti nel contatto
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]

            # Calcola la forza di contatto per questo punto
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            fz = abs(force[2])  # componente verticale

            if geom1_body == left_foot_id or geom2_body == left_foot_id:
                forces[0] += fz
            if geom1_body == right_foot_id or geom2_body == right_foot_id:
                forces[1] += fz

        return forces

    # ------------------------------------------------------------------
    # HELPER: conversione quaternione -> Eulero
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """
        Converte un quaternione MuJoCo [w, x, y, z] in angoli di Eulero [roll, pitch, yaw].
        Convenzione ZYX (yaw-pitch-roll).
        """
        w, x, y, z = quat

        # Roll (rotazione attorno all'asse X)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (rotazione attorno all'asse Y)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # Yaw (rotazione attorno all'asse Z)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    # ------------------------------------------------------------------
    # APPLICA PERTURBAZIONE ESTERNA
    # ------------------------------------------------------------------
    def apply_perturbation(self, force: np.ndarray, duration_steps: int = 10):
        """
        Applica una forza esterna al torso per un certo numero di step.
        Utile per testare la robustezza del controllore.

        Parametri
        ----------
        force          : [fx, fy, fz] forza in Newton nel frame mondo
        duration_steps : numero di step per cui la forza viene applicata
        """
        for _ in range(duration_steps):
            self.data.xfrc_applied[self.torso_id, :3] = force
            mujoco.mj_step(self.model, self.data)
            self._step_count += 1

        # Rimuove la forza dopo la durata
        self.data.xfrc_applied[self.torso_id, :3] = 0.0

    # ------------------------------------------------------------------
    # CHIUSURA
    # ------------------------------------------------------------------
    def close(self):
        """Chiude il viewer se aperto."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ----------------------------------------------------------------------
# Test rapido: carica il modello e stampa le info principali
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os

    xml_path = os.path.expanduser(
        "~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    )

    print("Caricamento modello G1 29 DOF...")
    env = G1Env(xml_path=xml_path, render=True)

    print(f"  Numero attuatori  : {env.num_joints}")
    print(f"  Dimensione qpos   : {env.num_qpos}")
    print(f"  Dimensione qvel   : {env.num_qvel}")
    print(f"  ID torso_link     : {env.torso_id}")

    # Reset e stampa osservazione iniziale
    obs = env.reset()
    print("\nOsservazione iniziale:")
    for k, v in obs.items():
        print(f"  {k:20s}: {v}")

    # Loop di test: azione zero (robot fermo) per 500 step
    print("\nEsecuzione 500 step con azione zero...")
    action = np.zeros(env.num_joints)
    step = 0
    while env.viewer is not None and env.viewer.is_running():
        obs, done, info = env.step(action)
        if done and step % 100 == 0:
            print(f"  Robot caduto, step {step}, altezza {obs['base_pos'][2]:.3f}m")
        step += 1
    print("Test completato.")
    # env.close()