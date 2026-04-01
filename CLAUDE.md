# ESE 6510 Drone Racing Project

## Coding Rules
- Only fill in TODO sections. Do NOT modify existing comments, non-TODO code, or repository structure.
- Preserve all original comments and docstrings exactly as they are.

## Project Overview
RL-based drone racing using NVIDIA Isaac Lab. Train a Crazyflie drone to race through gates on the "powerloop" track using PPO.

## Key Files (Submittable)
- `src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py` — PPO algorithm
- `src/third_parties/rsl_rl_local/rsl_rl/storage/rollout_storage.py` — Rollout storage / GAE
- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/agents/rsl_rl_ppo_cfg.py` — Network & hyperparams
- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py` — Rewards, obs, reset

## Track: Powerloop (7 gates)
Gate 3 and Gate 6 are the SAME physical gate traversed in OPPOSITE directions.
The powerloop maneuver is a vertical loop between gates 2→3.
The chicane is gates 5→6→0 (high-speed offset gates).

## Evaluation Rules
- **Metric:** Time to complete 3 laps
- **Initial pose:** x_local ∈ [-3.0, -0.5], y_local ∈ [-1.0, 1.0], starts from ground (z≈0.05)
- **Dynamics randomization:** TAs sample 3 parameters once from:
  - TWR: ±5%
  - Aero drag (xy/z): 0.5x–2.0x
  - PID gains (rp): kp/ki ±15%, kd ±30%
  - PID gains (yaw): kp/ki ±15%, kd ±30%
- **Gate direction:** Drone must NOT enter gate from wrong side (DQ if it does)
- **Cannot modify** `quadcopter_env.py` — TAs use their own version

## Training Commands
```bash
python scripts/rsl_rl/train_race.py --task Isaac-Quadcopter-Race-v0 --num_envs 8192 --max_iterations 8000 --headless --logger wandb
```

## Play Commands
```bash
python scripts/rsl_rl/play_race.py --task Isaac-Quadcopter-Race-v0 --num_envs 1 --load_run [YYYY-MM-DD_XX-XX-XX] --checkpoint best_model.pt --headless --video --video_length 800
```
