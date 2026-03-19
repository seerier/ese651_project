# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure

        num_gates = self.env._waypoints.shape[0]
        gate_half = self.env.cfg.gate_model.gate_side / 2.0  # 0.5 m

        # --- Gate-crossing detection via sign change of gate-frame x-coordinate ---
        # _pose_drone_wrt_gate is the drone position in the gate's local frame.
        # The gate's x-axis is its normal (approach direction). The drone starts on the
        # positive-x side and must fly through to the negative-x side to count as passed.
        curr_x = self.env._pose_drone_wrt_gate[:, 0]
        curr_y = self.env._pose_drone_wrt_gate[:, 1]
        curr_z = self.env._pose_drone_wrt_gate[:, 2]
        prev_x = self.env._prev_x_drone_wrt_gate

        # Crossed from positive to non-positive while within the gate aperture (±y, ±z)
        gate_passed = (
            (prev_x > 0.0)
            & (curr_x <= 0.0)
            & (torch.abs(curr_y) < gate_half + 0.1)
            & (torch.abs(curr_z) < gate_half + 0.1)
        )
        ids_gate_passed = torch.where(gate_passed)[0]

        # Advance waypoint index and update counters for envs that passed a gate
        if len(ids_gate_passed) > 0:
            self.env._n_gates_passed[ids_gate_passed] += 1
            self.env._idx_wp[ids_gate_passed] = (
                self.env._idx_wp[ids_gate_passed] + 1
            ) % num_gates
            # Update debug-vis target to next gate
            self.env._desired_pos_w[ids_gate_passed, :3] = (
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
            )
            # Recompute gate-relative pose using the NEW target gate so that
            # observations and the next progress step are consistent
            new_pose, _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_pos_w[ids_gate_passed],
            )
            self.env._pose_drone_wrt_gate[ids_gate_passed] = new_pose
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pose[:, 0]
            self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(
                new_pose, dim=1
            )

        # Update prev_x for envs that did NOT just pass a gate (used next timestep)
        self.env._prev_x_drone_wrt_gate[~gate_passed] = curr_x[~gate_passed]

        # --- Dense potential-based progress reward: reward for closing distance to gate ---
        dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        progress = (self.env._last_distance_to_goal - dist_to_gate).clamp(-5.0, 5.0)
        self.env._last_distance_to_goal = dist_to_gate.clone()

        # --- Velocity-toward-gate reward: encourages high racing speed ---
        # Rotate body-frame velocity to world frame via the rotation matrix
        rot_mat = matrix_from_quat(self.env._robot.data.root_quat_w)  # (N,3,3) body→world
        lin_vel_w = torch.bmm(
            rot_mat, self.env._robot.data.root_com_lin_vel_b.unsqueeze(-1)
        ).squeeze(-1)
        gate_pos_w  = self.env._waypoints[self.env._idx_wp, :3]
        drone_pos_w = self.env._robot.data.root_link_pos_w
        vec_to_gate = gate_pos_w - drone_pos_w
        dist_for_vel = torch.norm(vec_to_gate, dim=1, keepdim=True).clamp(min=1e-6)
        dir_to_gate = vec_to_gate / dist_for_vel               # unit vector toward gate
        vel_toward_gate = (lin_vel_w * dir_to_gate).sum(dim=1).clamp(-5.0, 10.0)

        # --- Crash detection: sustained contact force for > 100 timesteps ---
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "gate_pass":     gate_passed.float()  * self.env.rew['gate_pass_reward_scale'],
                "progress_goal": progress             * self.env.rew['progress_goal_reward_scale'],
                "velocity_gate": vel_toward_gate      * self.env.rew['velocity_gate_reward_scale'],
                "crash":         crashed.float()      * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations

        # Body-frame linear velocity: direct speed/direction signal for thrust control
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b   # (N, 3)

        # Body-frame angular velocity (body rates): essential for attitude stabilization
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b       # (N, 3)

        # Gravity vector projected into body frame: encodes roll and pitch tilt compactly.
        # R maps body→world, so g_body = R^T @ [0,0,-1] = -row-2 of R.
        rot_mat_obs = matrix_from_quat(self.env._robot.data.root_quat_w)  # (N, 3, 3)
        gravity_b = -rot_mat_obs[:, 2, :]                                  # (N, 3)

        # Current target gate: drone position in gate's local frame.
        # Gives direction and distance to gate in a frame-aligned coordinate system.
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate          # (N, 3)

        # Next gate look-ahead: lets the policy plan smooth racing lines through gates
        num_gates_obs = self.env._waypoints.shape[0]
        next_gate_idx = (self.env._idx_wp + 1) % num_gates_obs
        drone_pos_next_gate_frame, _ = subtract_frame_transforms(
            self.env._waypoints[next_gate_idx, :3],
            self.env._waypoints_quat[next_gate_idx, :],
            self.env._robot.data.root_link_pos_w,
        )  # (N, 3)

        # Previous policy outputs (CTBR): temporal context for smooth control
        prev_actions = self.env._previous_actions                     # (N, 4)

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                drone_lin_vel_b,            # 3  body-frame linear velocity
                drone_ang_vel_b,            # 3  body-frame angular velocity (body rates)
                gravity_b,                  # 3  gravity in body frame (roll/pitch tilt proxy)
                drone_pos_gate_frame,       # 3  drone position in current gate frame
                drone_pos_next_gate_frame,  # 3  drone position in next gate frame (look-ahead)
                prev_actions,               # 4  previous CTBR policy output
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.

        # ----------------------------------------------------------------
        # Domain randomization: re-sample dynamics each episode so the policy
        # learns to be robust to the perturbation ranges used in evaluation.
        # ----------------------------------------------------------------
        if self.cfg.is_train:
            # Thrust-to-weight ratio: ±5%
            self.env._thrust_to_weight[env_ids] = torch.empty(
                n_reset, device=self.device
            ).uniform_(self.env._twr_value * 0.95, self.env._twr_value * 1.05)

            # Aerodynamic drag coefficients: 0.5× – 2.0×
            k_xy = torch.empty(n_reset, device=self.device).uniform_(
                self.env._k_aero_xy_value * 0.5, self.env._k_aero_xy_value * 2.0
            )
            self.env._K_aero[env_ids, 0] = k_xy
            self.env._K_aero[env_ids, 1] = k_xy
            self.env._K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                self.env._k_aero_z_value * 0.5, self.env._k_aero_z_value * 2.0
            )

            # PID gains – roll/pitch: kp/ki ±15%, kd ±30%
            kp_rp = torch.empty(n_reset, device=self.device).uniform_(
                self.env._kp_omega_rp_value * 0.85, self.env._kp_omega_rp_value * 1.15
            )
            ki_rp = torch.empty(n_reset, device=self.device).uniform_(
                self.env._ki_omega_rp_value * 0.85, self.env._ki_omega_rp_value * 1.15
            )
            kd_rp = torch.empty(n_reset, device=self.device).uniform_(
                self.env._kd_omega_rp_value * 0.70, self.env._kd_omega_rp_value * 1.30
            )
            self.env._kp_omega[env_ids, 0] = kp_rp
            self.env._kp_omega[env_ids, 1] = kp_rp
            self.env._ki_omega[env_ids, 0] = ki_rp
            self.env._ki_omega[env_ids, 1] = ki_rp
            self.env._kd_omega[env_ids, 0] = kd_rp
            self.env._kd_omega[env_ids, 1] = kd_rp

            # PID gains – yaw: kp/ki ±15%, kd ±30%
            kp_y = torch.empty(n_reset, device=self.device).uniform_(
                self.env._kp_omega_y_value * 0.85, self.env._kp_omega_y_value * 1.15
            )
            ki_y = torch.empty(n_reset, device=self.device).uniform_(
                self.env._ki_omega_y_value * 0.85, self.env._ki_omega_y_value * 1.15
            )
            kd_y = torch.empty(n_reset, device=self.device).uniform_(
                self.env._kd_omega_y_value * 0.70, self.env._kd_omega_y_value * 1.30
            )
            self.env._kp_omega[env_ids, 2] = kp_y
            self.env._ki_omega[env_ids, 2] = ki_y
            self.env._kd_omega[env_ids, 2] = kd_y

        # ----------------------------------------------------------------
        # Curriculum starting positions: randomly sample ANY gate on the track.
        # This immediately exposes the policy to all gates and approach angles,
        # dramatically accelerating learning compared to always starting at gate 0.
        # ----------------------------------------------------------------
        num_gates_reset = self.env._waypoints.shape[0]
        waypoint_indices = torch.randint(
            0, num_gates_reset, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
        )

        # Gate world pose
        x0_wp = self.env._waypoints[waypoint_indices, 0]
        y0_wp = self.env._waypoints[waypoint_indices, 1]
        z_wp  = self.env._waypoints[waypoint_indices, 2]
        theta = self.env._waypoints[waypoint_indices, -1]

        # Randomized approach offset in gate local frame:
        #   x_local > 0  → drone in front of gate (positive gate-frame x)
        #   y_local      → lateral spread within the gate corridor
        #   z_local      → height variation around gate centre
        approach_d = torch.empty(n_reset, device=self.device).uniform_(1.2, 3.5)
        lat_offset = torch.empty(n_reset, device=self.device).uniform_(-0.35, 0.35)
        z_offset   = torch.empty(n_reset, device=self.device).uniform_(-0.25, 0.25)

        # Rotate from gate local frame to world frame
        # Gate x-axis in world = [cos(theta), sin(theta), 0]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx_w = cos_theta * approach_d - sin_theta * lat_offset
        dy_w = sin_theta * approach_d + cos_theta * lat_offset

        initial_x = x0_wp + dx_w
        initial_y = y0_wp + dy_w
        initial_z = (z_wp + z_offset).clamp(0.15, self.env.cfg.max_altitude - 0.2)

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z
        default_root_state[:, 7:] = 0.0  # zero initial velocity for clean episode start

        # Point drone toward chosen gate + small yaw noise for orientation robustness
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + torch.empty(n_reset, device=self.device).uniform_(-0.20, 0.20),
        )
        default_root_state[:, 3:7] = quat

        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0