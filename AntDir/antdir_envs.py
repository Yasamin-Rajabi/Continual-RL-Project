"""Ant direction tasks (a Gymnasium port of the PEARL/oyster reward).

No direction/task ID is appended to the observation. This is deliberately a
NEW benchmark, not a replacement for ../ant (the calibrated velocity tasks).
Reward: projected torso velocity + 1 - .5||a||^2 - .0005||clipped contacts||^2.
27 proprioceptive coordinates + contact forces; 200-step time limit.
The success/heading_error diagnostics are ours, not a canonical AntDir metric.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv


class AntDirectionEnv(AntEnv):
    def __init__(self, direction: float, render_mode=None):
        self.direction = float(direction)
        super().__init__(render_mode=render_mode, use_contact_forces=True,
                         exclude_current_positions_from_observation=True,
                         healthy_z_range=(0.2, 1.0), terminate_when_unhealthy=True,
                         ctrl_cost_weight=0.5, contact_cost_weight=5e-4)

    def step(self, action):
        xy_before = self.get_body_com('torso')[:2].copy()
        self.do_simulation(action, self.frame_skip)
        velocity = (self.get_body_com('torso')[:2] - xy_before) / self.dt
        target = np.array([np.cos(self.direction), np.sin(self.direction)])
        projected = float(velocity @ target)
        ctrl = 0.5 * float(np.square(action).sum())
        contact = 5e-4 * float(np.square(np.clip(self.data.cfrc_ext, -1, 1)).sum())
        state = self.state_vector()
        healthy = bool(np.isfinite(state).all() and 0.2 <= state[2] <= 1.0)
        speed = float(np.linalg.norm(velocity))
        error = (float(np.arccos(np.clip(projected / speed, -1.0, 1.0)))
                 if speed > 1e-8 else float(np.pi))
        info = dict(reward_forward=projected, reward_ctrl=-ctrl,
                    reward_contact=-contact, reward_survive=1.0,
                    x_velocity=float(velocity[0]), y_velocity=float(velocity[1]),
                    projected_velocity=projected, heading_error=error,
                    # Alias needed by the existing continuous-control log/plot code.
                    velocity_error=error,
                    success=float(healthy and projected >= 0.2 and error <= np.pi/12))
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), projected + 1.0 - ctrl - contact, not healthy, False, info


def make_env(direction: float, render=False):
    return gym.wrappers.TimeLimit(
        AntDirectionEnv(direction, render_mode='human' if render else None),
        max_episode_steps=200)
