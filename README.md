# G1 Humanoid Controller — Isaac Lab

Isaac Lab extension that trains a bipedal walking/standing controller for the Unitree G1 (29-DOF) humanoid using PPO (RSL-RL).


!

## Available tasks

| Task ID | Description |
|---|---|
| `Template-G1_29dof-Controller-v0` | Full 29-DOF G1, forward walking, target velocity 0.4 m/s on x, self-collisions enabled. |
| `Template-G1_gait-Controller-v0` | Standing/balance task, velocity command fixed at 0 (used to learn stability before walking). |

## Project structure

```
source/G1_controller/
└── G1_controller/
    └── tasks/manager_based/g1_controller/
        ├── g1_29dof_env_cfg.py   # walking env: scene, rewards, terminations
        ├── g1_gait_env_cfg.py    # standing/stability env
        ├── robots/usd_cfg.py     # G1 articulation/actuator config
        ├── agents/rsl_rl_ppo_cfg.py  # PPO hyperparameters
        └── mdp/rewards.py        # custom reward functions
scripts/
├── list_envs.py        # list registered tasks
├── zero_agent.py        # sanity check: zero actions
├── random_agent.py      # sanity check: random actions
└── rsl_rl/
    ├── train.py
    ├── play.py
    └── cli_args.py
```

## Docker setup

The `docker/Dockerfile` builds on `nvcr.io/nvidia/isaac-sim:5.1.0` and bakes Isaac Lab (`isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl`) into the image. `docker-compose.yml` (in `docker/`) builds that image and mounts `~/docker/isaac-sim/documents` to `/root/Documents`.

1. **Get Isaac Lab source** (build-time dependency, copied into the image):
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git docker/IsaacLab
   ```
2. **Place this project where the volume mounts it**, so it's visible inside the container:
   ```bash
   mkdir -p ~/docker/isaac-sim/documents
   mv /path/to/G1_controller ~/docker/isaac-sim/documents/G1_controller
   ```
3. **Build and start the container**:
   ```bash
   cd ~/docker/isaac-sim/documents/G1_controller/docker
   docker compose up -d --build
   ```
4. **Enter the container** and go to the project (mounted at `/root/Documents`):
   ```bash
   docker exec -it isaac-sim-container bash
   cd /root/Documents/G1_controller
   ```

All commands below run **inside the container**, from this path. (GUI rendering needs X11 on the host: `xhost +local:` before step 3 if you hit display errors.)

## Installation

Isaac Lab core packages are already installed in the image. Only this project's extension needs installing:
```bash
/isaac-sim/python.sh -m pip install -e source/G1_controller
```
Verify installation:
```bash
/isaac-sim/python.sh scripts/list_envs.py
```
(Optional) sanity-check an env before training:
```bash
/isaac-sim/python.sh scripts/zero_agent.py --task Template-G1_29dof-Controller-v0
/isaac-sim/python.sh scripts/random_agent.py --task Template-G1_29dof-Controller-v0
```

## Training

```bash
# start training
/isaac-sim/python.sh scripts/rsl_rl/train.py --task Template-G1_29dof-Controller-v0

# resume / extend a previous run
/isaac-sim/python.sh scripts/rsl_rl/train.py --task Template-G1_29dof-Controller-v0 --resume --max_iterations 500
```

Replace the task with `Template-G1_gait-Controller-v0` to train the standing/balance policy instead.

PPO settings (network size, learning rate, iterations, etc.) are defined in `agents/rsl_rl_ppo_cfg.py`. Logs are written to `logs/rsl_rl/<experiment_name>/<run_id>/`.

Monitor training:
```bash
/isaac-sim/python.sh -m tensorboard.main --logdir=logs
```

## Using a trained policy

```bash
/isaac-sim/python.sh scripts/rsl_rl/play.py \
    --task Template-G1_29dof-Controller-v0 \
    --load_run <run_id_folder> \
    --num_envs 1
```

`--load_run` is the timestamped folder name under `logs/rsl_rl/<experiment_name>/` (omit it to load the most recent run). `play.py` also exports the policy to `policy.pt` (JIT) and `policy.onnx` for deployment outside Isaac Lab.

## Reward design (summary)

Rewards/penalties are implemented in `mdp/rewards.py` and combined per-task in the `*_env_cfg.py` files:
- **Task reward**: velocity tracking (`forward_velocity`) and an alive bonus.
- **Stability**: penalties on vertical/angular base velocity, orientation tilt, and base height deviation.
- **Gait quality**: foot air-time, contact alternation, foot-flat/landing, and leg-symmetry rewards.
- **Regularization**: joint velocity/torque penalties and action-rate smoothing to avoid jittery motion.

Termination conditions: episode timeout, excessive base tilt, and base height below a minimum threshold.