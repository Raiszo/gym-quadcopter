from dataclasses import dataclass
from typing import Optional
import gym
from gym import spaces
import numpy as np
from gym.envs.classic_control import rendering

g = 9.81

@dataclass
class EnvProps:
    initial_state: np.ndarray
    """Vector of size 6: x, z, phi, and the derivatives(3)"""
    objective: np.ndarray
    """Target state"""
    dt: float
    """timestep"""

class QuadcopterEnv(gym.Env):
    def __init__(self, props: EnvProps) -> None:
        super().__init__()
        self.props = props

        self.mass = 0.5         # mass [kg]
        self.Ixx = 0.00232      # inertia in X-Z plane
        self.arm_length = 0.1   # arm length [m] whole length would be x2
        self.arm_height = 0.02  # for visualization

        self.state: np.ndarray
        self.initial_state = props.initial_state
        self.objective = props.objective

    def step(self, action: np.ndarray):
        """action of size 2 the firts is the right one"""
        u1, u2 = action

        F = u1 + u2
        M = (u1 - u2) * self.arm_length

        _, _, phi, x_dot, z_dot, phi_dot = self.state
        S_dot = np.array([
            x_dot,
            z_dot,
            phi_dot,
            -F * np.sin(phi) / self.mass,
            F * np.cos(phi) /  self.mass - g,
            M / self.Ixx,
        ])

        self.state = S_dot * self.props.dt + self.state

    def reset(self):
        self.state = self.initial_state

    def render(self, mode='human', close=False):
        pass
