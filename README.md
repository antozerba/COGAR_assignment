Subgroup D1: G1 EDU Humanoid Locomotion

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

# Stating point:
- The robot Unitree G1 29 degrees of freedom (DOF) (https://github.com/unitreerobotics/unitree_mujoco).
- 3D LIDAR (LIVOX-MID360) + Depth Camera Intel RealSense (D435i)
