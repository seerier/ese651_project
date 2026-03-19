# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Longer rollouts give better GAE estimates; 48 steps ≈ ~1 s at 50 Hz policy rate
    num_steps_per_env = 48
    max_iterations = 5000
    save_interval = 25
    experiment_name = "quadcopter_direct"
    empirical_normalization = False
    wandb_project = "ese651_quadcopter"  # Wandb project name for logging
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Wider actor: 19-dim obs → 256 → 256 → 128 → 4-dim action
        actor_hidden_dims=[256, 256, 128],
        # Deeper critic: better value estimates drive better advantage signals
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
        min_std=0.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Positive entropy bonus encourages exploration and diverse racing lines
        entropy_coef=0.005,
        num_learning_epochs=5,
        # 8 mini-batches with 48 steps × 8192 envs → ~49k samples each, stable gradient
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        # Higher gamma for longer-horizon lap-time optimisation (~2.8 s half-life at 50 Hz)
        gamma=0.995,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
