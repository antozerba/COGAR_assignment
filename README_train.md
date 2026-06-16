# G1 Humanoid Controller - Isaac Lab

Assignment 8: G1 EDU Bipedal Walking Controller (SIMULATION)
Student id:

What to do: Implement and evaluate a stable walking controller for the Unitree G1 EDU humanoid robot in MuJoCo.
1) Set up the G1 EDU robot model in MuJoCo using available Unitree simulation resources.
2) Verify correct simulation of: joint kinematics, foot-ground contacts, base pose and joint state feedback
3) Implement or adapt a walking controller, for example: a state-machine-based controller, a trajectory tracking controller, a reinforcement-learning-based baseline using existing Unitree repositories
4) Tune the controller for stable walking on flat terrain.
5) Test robustness under: different walking speeds, small external perturbations, different initial conditions
6) Evaluate locomotion performance using quantitative metrics such as: distance traveled, average velocity, number of falls, orientation stability adn recovery behavior after perturbations
7) Compare at least two controller settings or two locomotion approaches.


Software needed: MuJoCo, Python, unitree_mujoco, unitree_rl_mjlab, plotting and logging tools (NumPy, Matplotlib, Pandas)
Research needed: Bipedal locomotion for humanoid robots, walking control theory, contact-aware control, reinforcement learning for humanoid locomotion, Unitree G1 EDU documentation
Deliverables: Working G1 EDU walking simulation in MuJoCo, locomotion controller implementation, benchmark experiments, stability analysis report, demo videos

---

## Reward Function

L'andatura e il comportamento del robot sono governati da una funzione di reward combinata all'interno del file `g1_controller_env_cfg.py`. La funzione bilancia obiettivi primari (premi positivi) e vincoli di regolarità fisica/cinematica (penalità negative).

Di seguito viene dettagliato il funzionamento di ogni singolo termine attivo nel sistema:

### 1. Premi Positivi (Obiettivi)

*   **`forward_velocity` (Peso: +2.0)**
    *   *Funzione:* `track_lin_vel_xy_exp` con tolleranza `std: 0.25`.
    *   *Descrizione:* È il motore principale dell'addestramento. Usa una curva a campana esponenziale $\exp(-x^2 / \sigma^2)$ per premiare il robot quando la sua velocità reale si allinea a quella richiesta dal comando. Essendo il valore positivo più alto, la policy ottimizza aggressivamente questo termine.
*   **`alive` (Peso: +0.5)**
    *   *Funzione:* `is_alive`.
    *   *Descrizione:* Premio costante erogato ad ogni step in cui il robot non cade. Impedisce il "suicidio" dell'agente nelle prime fasi di esplorazione, spingendolo a lottare per mantenere l'equilibrio.

### 2. Penalità Negative (Vincoli Dinamici e di Postura)

*   **`flat_orientation` (Peso: -2.0)**
    *   *Funzione:* `flat_orientation_l1`.
    *   *Descrizione:* Penalizza la deviazione del vettore di gravità proiettato nel sistema di riferimento locale del robot rispetto alla verticale globale. Serve a mantenere il torso eretto ed evitare che si inclini eccessivamente.
*   **`lin_vel_z` (Peso: -0.5)**
    *   *Funzione:* `lin_vel_z_l2`.
    *   *Descrizione:* Penalizza i movimenti e le velocità lungo l'asse verticale ($Z$). Impedisce all'umanoide di saltellare o rimbalzare vistosamente (effetto canguro), stabilizzando la camminata su un piano uniforme.
*   **`ang_vel_xy` (Peso: -0.05)**
    *   *Funzione:* `ang_vel_xy_l2`.
    *   *Descrizione:* Penalizza le velocità angolari di rollio (roll) e beccheggio (pitch) della base, attutendo le oscillazioni violente del bacino ad ogni passo.
*   **`action_rate` (Peso: -0.01)**
    *   *Funzione:* `action_rate_l2`.
    *   *Descrizione:* Penalizza le variazioni brusche tra comandi consecutivi (derivata dell'azione). Forzando la fluidità (*smoothness*), elimina le vibrazioni ad alta frequenza che distruggerebbero i giunti del robot reale.
*   **`joint_vel` (Peso: -0.001)**
    *   *Funzione:* `joint_vel_l1`.
    *   *Descrizione:* Penalizza il valore assoluto della velocità dei motori per ottimizzare l'efficienza energetica, riducendo i movimenti frenetici o inutili degli arti.

### 3. Condizione di Terminazione (Fallimento)

*   **`terminating` (Peso: -10.0)**
    *   *Funzione:* `is_terminated`.
    *   *Descrizione:* Punizione massima applicata istantaneamente se il robot cade o tocca il suolo con il torso, interrompendo bruscamente l'episodio. Agisce come barriera critica per costringere l'algoritmo a dare priorità assoluta alla stabilità.

---

## Struttura del Workspace

```rc
.
├── pyproject.toml         # Configurazione dei percorsi extra per l'autocompletamento di VS Code
├── scripts/               
│   └── rsl_rl/            
│       ├── train.py       # Script per avviare o riprendere l'addestramento
│       └── play.py        # Script per visualizzare graficamente la policy appresa
└── source/
    └── G1_controller/
        └── G1_controller/
            └── tasks/
                └── manager_based/
                    └── g1_controller/
                        ├── g1_controller_env_cfg.py  # File principale della scena e dei reward
                        ├── agents/
                        │   └── rsl_rl_ppo_cfg.py     # Configurazione dell'algoritmo PPO
                        └── mdp/
                            └── rewards.py            # Funzioni matematiche di reward customz
```

## Running CMD

- Move to the project folder dir 
``` bash
cd /root/Documents/G1_controller/
```

- Training
``` bash
/isaac-sim/python.sh scripts/rsl_rl/train.py --task  Template-G1-Controller-v0
```

-  Extend training previously done
``` bash
/isaac-sim/python.sh scripts/rsl_rl/train.py --task Template-G1-Controller-v0 --resume --max_iterations 500
```
-  Play
``` bash
/isaac-sim/python.sh scripts/rsl_rl/play.py --task Template-G1-Controller-v0 --load_run 2026-05-23_14-43-38 --num_envs 1
```
- Tensorboard
``` bash
/isaac-sim/python.sh -m tensorboard.main --logdir=logs
```

