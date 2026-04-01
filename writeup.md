# Drone Racing Policy: Strategy Write-Up

## 1. Overview

We trained a PPO-based drone racing policy on the Powerloop track (7 gates, 3 laps) using NVIDIA Isaac Lab. Our strategy centers on four pillars: (1) a multi-component reward structure that balances gate-seeking speed with safety, (2) a rich 28-dimensional observation space enabling multi-gate planning, (3) a curriculum-based reset strategy including ground starts, and (4) aggressive domain randomization matching the evaluation perturbation ranges.

## 2. Reward Structure

Our reward function combines eight components, each tuned to shape specific behaviors:

**Primary signals:**
- **Gate passage (+15.0):** Large sparse bonus on each correct gate traversal. This is the dominant training signal driving gate-seeking behavior.
- **Progress toward gate (+2.0):** Dense potential-based shaping that rewards reducing distance to the current target gate. Provides gradient between sparse gate-pass events.
- **Velocity toward gate (+0.5):** Rewards the component of world-frame velocity directed at the current gate, directly incentivizing speed over cautious hovering.
- **Gate speed bonus (+1.0):** At the moment of gate passage, rewards velocity aligned with the gate normal vector. This encourages fast, clean traversals rather than slow drift-throughs.

**Penalties:**
- **Wrong-side entry (-5.0):** Detects when the drone crosses a gate plane from the wrong direction (negative-x to positive-x in gate frame). This is critical for the shared gate 3/6, where the same physical gate must be traversed in opposite directions on different laps segments. Without this penalty, the policy learns a shortcut that would result in disqualification.
- **Time penalty (-0.05/step):** A constant per-timestep cost that creates persistent pressure toward faster lap completion. Over a 30-second episode at 50 Hz, this amounts to -75 total, balanced against ~315 from 21 gate passes (3 laps x 7 gates x 15.0).
- **Crash penalty (-0.5/step):** Penalizes sustained contact forces (>100 timesteps), discouraging collision with gates and ground.
- **Angular velocity penalty (-0.01):** Penalizes excessive body rotation rates, promoting smooth flight and reducing energy-wasting oscillations.
- **Death cost (-15.0):** Terminal penalty on fatal termination (altitude violation, sustained crash, off-track).

## 3. Observation Space (28 dimensions)

We designed observations to give the policy sufficient information for multi-gate trajectory planning:

| Component | Dims | Frame | Purpose |
|-----------|------|-------|---------|
| Linear velocity | 3 | Body | Speed and direction for thrust control |
| Angular velocity | 3 | Body | Attitude stabilization |
| Gravity vector | 3 | Body | Roll/pitch tilt encoding |
| Current gate position | 3 | Gate | Direction and distance to target |
| Next gate position (gate+1) | 3 | Gate+1 | Racing line planning |
| Gate+2 position | 3 | Gate+2 | Powerloop/chicane setup |
| Gate normal direction | 3 | Body | Gate traversal direction (critical for gate 3/6) |
| Distance to gate | 1 | Scalar | Value function estimation |
| Height | 1 | World | Ground avoidance, takeoff |
| Lap progress | 1 | Scalar | Multi-lap value estimation |
| Previous actions | 4 | N/A | Temporal smoothness |

The gate normal in body frame is particularly important: it tells the policy which direction to fly through the current gate. For gates 3 and 6 (same physical gate, opposite traversal directions), this observation disambiguates the correct approach.

The two-gate look-ahead (gate+1 and gate+2) enables the policy to plan smooth racing lines. This is critical for the powerloop maneuver (gates 2 to 3) where the drone must set up a vertical loop, and for the chicane (gates 5 to 6 to 0) where rapid direction changes are needed.

## 4. Reset Strategy and Curriculum

**Curriculum starting positions:** During training, episodes begin at random gates with approach offsets that grow with training progress:
- Early training (iteration 0): approach distance 0.5-1.5m (close, easy gate passes)
- Late training (iteration 2000+): approach distance 1.2-3.5m (full difficulty)

This curriculum accelerates early learning by providing frequent gate-pass rewards, then gradually increases difficulty for robustness.

**Ground starts (20% of resets):** To match the TA evaluation condition where the drone starts on the ground (z=0.05) with randomized position near gate 0, we include ground starts in 20% of training resets. The x_local and y_local are sampled from the same distributions as evaluation: x_local in [-3.0, -0.5], y_local in [-1.0, 1.0].

**Velocity noise:** Non-ground resets include small initial velocity perturbations (uniform [-0.3, 0.3] m/s) to improve robustness to disturbances during flight.

## 5. Domain Randomization

Every training episode randomizes dynamics parameters matching the evaluation ranges:
- Thrust-to-weight ratio: +/-5% of nominal
- Aerodynamic drag (xy and z): 0.5x to 2.0x nominal
- PID gains for roll/pitch: kp/ki +/-15%, kd +/-30%
- PID gains for yaw: kp/ki +/-15%, kd +/-30%

This ensures the policy is robust to the 3 randomly sampled parameters used in TA evaluation.

## 6. PPO Hyperparameters

- **Network:** Actor [256, 256, 128], Critic [512, 512, 256, 128], ELU activation
- **Rollout length:** 64 steps (~1.3s at 50 Hz) for improved GAE estimates
- **Discount factor:** gamma = 0.997 (longer horizon for 3-lap optimization)
- **Learning rate:** 3e-4 with adaptive KL-based scheduling (target KL = 0.008)
- **Entropy coefficient:** 0.008 (encourages exploration of diverse racing lines)
- **Mini-batches:** 8 per update, 5 epochs per rollout
- **Training budget:** 8000 iterations with 8192 parallel environments

## 7. Racing Strategy

The policy develops an aggressive racing style: it accelerates toward each gate, aims for center-of-gate traversals at high speed (rewarded by the gate speed bonus), and immediately redirects toward the next gate. The two-gate look-ahead enables anticipatory turns, particularly important for:

1. **Powerloop (gates 2-3):** The policy learns to approach gate 2 with upward momentum, execute a climbing arc, and dive through gate 3 from the correct direction.
2. **Chicane (gates 5-6-0):** Rapid lateral alternation through offset gates at low altitude.
3. **Gate 3/6 disambiguation:** The gate normal observation ensures the policy never confuses the approach direction for this shared physical gate.
