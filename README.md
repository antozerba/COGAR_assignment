# G1 Humanoid Controller — Isaac Lab

Isaac Lab extension that trains a bipedal walking/standing controller for the Unitree G1 (29-DOF) humanoid using PPO (RSL-RL).


## Available tasks

| Task ID | Description |
|---|---|
| `Template-G1_29dof-Controller-v0` | Full 29-DOF G1, forward walking, self-collisions enabled. |
| `Template-G1_gait-Controller-v0` | Full 29-DOF G1, Gai shift implementation, different phases (standing-walking-running) |
| `Template-G1_mujoco-Controller-v0` | Full 29-DOF G1, all directions locomotive task based on mujoco weights |

## Project structure

Below is the directory layout of the `G1_controller` repository along with a brief description of each component:

```text
.
├── docker/                      # Containerization configuration files
│   ├── Dockerfile               # Recipes to build the controller's Docker image
│   └── docker-compose.yaml      # Multi-container orchestration (e.g., managing simulation/headless runs)
├── logs/                        # Training logs, checkpoints, and evaluation metrics
│   └── rsl_rl/                  
│       ├── G1_29dof_controller_ppo/   
│       ├── G1_gait_controller_ppo/    
│       └── G1_mujoco_controller_ppo/  
│           └── <timestamp>/     # Individual training sessions containing:
│               ├── events.out.* # TensorBoard event logs for training visualization
│               ├── exported/    # Final deployment-ready formats (policy.pt, policy.onnx)
│               ├── params/      # Configuration snapshots (agent.yaml, env.yaml) used for that run
│               └── videos/      
├── scripts/                     # Executable scripts to run, train, or test the framework
│   ├── list_envs.py             
│   ├── random_agent.py          
│   ├── zero_agent.py            
│   └── rsl_rl/                  
│       ├── cli_args.py          
│       ├── train.py             # Main entrypoint to start reinforcement learning (PPO) training
│       └── play.py              # Script to visualize and evaluate trained checkpoints
├── source/                      
│   └── G1_controller/           
│       ├── config/              # Simulator integration configurations (e.g., Omniverse Isaac extension)
│       ├── docs/                # Project local documentation and changelogs
│       └── G1_controller/       # Environment logic, tasks, and state machine configurations
│           ├── tasks/           
│           │   └── manager_based/
│           │       └── g1_controller/
│           │           ├── agents/         # PPO Hyperparameters configuration 
│           │           ├── mdp/            # Markov Decision Process functions 
│           │           ├── robots/         # Robot-specific USD and mesh asset loaders
│           │           ├── g1_29dof_env_cfg.py   
│           │           ├── g1_gait_env_cfg.py    
│           │           └── g1_mujoco_env_cfg.py  
│           └── ui_extension_example.py 
├── pyproject.toml               
├── setup.py                     
└── README.md                    

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
   docker compose up -d 
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

## Training

```bash
# start training
/isaac-sim/python.sh scripts/rsl_rl/train.py --task Template-G1_29dof-Controller-v0

# resume / extend a previous run
/isaac-sim/python.sh scripts/rsl_rl/train.py --task Template-G1_29dof-Controller-v0 --resume --max_iterations 500
```

Replace the task with `Template-G1_mujoco-Controller-v0` to train for every directions.

PPO settings (network size, learning rate, iterations, etc.) are defined in `agents/rsl_rl_ppo_cfg.py`. 
Logs are written to `logs/rsl_rl/<experiment_name>/<run_id>/`.

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

Example:
``` bash
/isaac-sim/python.sh scripts/rsl_rl/play.py \
    --task Template-G1_mujoco-Controller-v0 \
    --load_run 2026-06-21_17-28-35 \
    --video \
    --video_length 500 \
    --num_envs 1
```


### TroubleShooting
- Before running the command make sure to allow X11 access to docker wiht `xhost +` outsidde the container
- Every time you want to visualize the play.py of a different task make sure to change the correct `experiment_name =""`  in the `agents/rsl_rl_ppo_cfg.py` script.
- In order to change the input velociy command,  in the env configuration file change the var `lin_vel_x=(0.5, 0.5)` in the `CommandsCfgPlay` class in mujoco_env
