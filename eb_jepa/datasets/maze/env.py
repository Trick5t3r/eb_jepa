"""Maze gym environment for MPPI/CEM planning evaluation.

Matches the surface of ``two_rooms.env.DotWall`` so the existing
``GCAgent`` / ``MPPIPlanner`` / ``CEMPlanner`` infrastructure runs unchanged.

Eval scenario (corner-to-corner):
  - On ``reset()`` a fresh random DFS maze is generated (retried until A* path
    >= ``min_path_length``), agent placed at the top-left path cell, goal at
    the bottom-right path cell.
  - Planner sees a 2-channel observation: Gaussian dot + maze wall mask.
  - ``step()`` accepts a continuous 2D action; we round it to the closest
    cardinal direction and move one cell — staying in place if the target
    cell is a wall.
  - Success threshold = 1 cell (cell_size pixels in pixel space).
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from eb_jepa.datasets.maze.maze_dataset import (
    MazeDatasetConfig,
    cell_to_pixel,
    generate_path_and_actions,
    render_dot,
    render_wall_mask,
)
from eb_jepa.datasets.maze.maze_solver import DIRECTIONS
from eb_jepa.datasets.maze.normalizer import MazeNormalizer

InfoType = Dict[str, Any]
ObsType = torch.Tensor


class MazeEnv(gym.Env):
    def __init__(
        self,
        config: MazeDatasetConfig,
        rng: Optional[np.random.Generator] = None,
        n_steps: int = 200,
        n_allowed_steps: int = 200,
        max_step_norm: float = 1.5,  # in cell units; planner output magnitude
        normalize: bool = True,
        **_unused,  # tolerate two_rooms-specific eval kwargs
    ):
        super().__init__()
        self.config = config
        self.maze_height = config.maze_height
        self.maze_width = config.maze_width
        self.cell_size = config.cell_size
        self.img_size = config.img_size
        self.agent_std = config.agent_std

        self.n_steps = n_steps
        self.n_allowed_steps = n_allowed_steps
        self.rng = rng or np.random.default_rng()

        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Planner outputs continuous 2D vectors in pixel-space (magnitude ~ cell_size)
        self.action_space = gym.spaces.Box(
            low=-max_step_norm * self.cell_size,
            high=max_step_norm * self.cell_size,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, self.img_size, self.img_size),
            dtype=np.float32,
        )

        self.normalize = normalize
        if normalize:
            self.normalizer = MazeNormalizer(img_size=self.img_size)
        else:
            self.normalizer = None

        # Episode state (populated by reset)
        self.maze_grid: Optional[torch.Tensor] = None
        self.wall_img: Optional[torch.Tensor] = None
        self.agent_cell: Optional[np.ndarray] = None
        self.goal_cell: Optional[np.ndarray] = None
        self.dot_position: Optional[torch.Tensor] = None
        self.target_position: Optional[torch.Tensor] = None
        self.position_history = None

    @property
    def np_random(self):
        return self.rng

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, location=None) -> Tuple[ObsType, InfoType]:
        # Generate a maze with a sufficiently long corner-to-corner path
        maze_np, _, _, start_cell, goal_cell = generate_path_and_actions(
            self.config, rng=self.rng
        )
        self.maze_grid = torch.from_numpy(maze_np.astype(np.int64)).to(self.device)
        self.wall_img = render_wall_mask(self.maze_grid, self.cell_size)  # (img, img)

        self.agent_cell = start_cell.copy()
        self.goal_cell = goal_cell.copy()

        self.dot_position = torch.tensor(
            cell_to_pixel(self.agent_cell, self.cell_size),
            device=self.device,
            dtype=torch.float32,
        )
        self.target_position = torch.tensor(
            cell_to_pixel(self.goal_cell, self.cell_size),
            device=self.device,
            dtype=torch.float32,
        )

        if location is not None:
            # Allow override for replay-style evaluation
            self.dot_position = location

        self.position_history = [self.dot_position]
        obs = self._render_dot_and_wall()
        info = self._build_info()
        return obs, info

    def step(self, action: np.array) -> Tuple[ObsType, float, bool, bool, InfoType]:
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action, dtype=np.float32)

        # Discretise to cardinal direction (max-magnitude axis)
        dr_cont, dc_cont = float(action_np[0]), float(action_np[1])
        if abs(dr_cont) >= abs(dc_cont):
            dr, dc = (1 if dr_cont > 0 else -1) if dr_cont != 0 else 0, 0
        else:
            dr, dc = 0, (1 if dc_cont > 0 else -1) if dc_cont != 0 else 0

        # Apply move if target cell is a path
        nr, nc = int(self.agent_cell[0] + dr), int(self.agent_cell[1] + dc)
        moved = False
        if (
            0 <= nr < self.maze_height
            and 0 <= nc < self.maze_width
            and int(self.maze_grid[nr, nc].item()) == 1
        ):
            self.agent_cell = np.array([nr, nc], dtype=np.int32)
            moved = True

        self.dot_position = torch.tensor(
            cell_to_pixel(self.agent_cell, self.cell_size),
            device=self.device,
            dtype=torch.float32,
        )
        self.position_history.append(self.dot_position)

        obs = self._render_dot_and_wall()
        # Done when at goal cell
        done = bool(
            self.agent_cell[0] == self.goal_cell[0]
            and self.agent_cell[1] == self.goal_cell[1]
        )
        truncated = len(self.position_history) >= self.n_allowed_steps
        reward = 1.0 if done else 0.0
        info = self._build_info()
        info["moved"] = moved
        return obs, reward, done, truncated, info

    def step_multiple(self, actions: np.ndarray):
        obs_, rew_, done_, trunc_, info_ = [], [], [], [], []
        for t in range(actions.shape[0]):
            o, r, d, tr, i = self.step(actions[t])
            obs_.append(o)
            rew_.append(r)
            done_.append(d)
            trunc_.append(tr)
            info_.append(i)
        return obs_, rew_, done_, trunc_, info_

    def eval_state(self, goal_dot_position, curr_dot_position, succes_treshold=None):
        """Success when agent is at the goal cell (pixel-space distance < cell_size)."""
        if succes_treshold is None:
            succes_treshold = self.cell_size + 0.5
        if isinstance(goal_dot_position, torch.Tensor):
            goal_dot_position = goal_dot_position.detach().cpu().numpy()
        if isinstance(curr_dot_position, torch.Tensor):
            curr_dot_position = curr_dot_position.detach().cpu().numpy()
        state_dist = float(np.linalg.norm(goal_dot_position - curr_dot_position))
        return {
            "success": state_dist < succes_treshold,
            "state_dist": state_dist,
        }

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def render(self):
        return self._render_dot_and_wall()

    def get_target_obs(self):
        return self._render_dot_at(self.target_position)

    def _build_info(self) -> InfoType:
        return {
            "dot_position": self.dot_position,
            "target_position": self.target_position,
            "target_obs": self.get_target_obs(),
        }

    def _render_dot_at(self, position: torch.Tensor) -> torch.Tensor:
        dot = render_dot(
            position,
            self.img_size,
            self.agent_std,
            device=self.device,
        )
        return torch.stack([dot, self.wall_img], dim=0)

    def _render_dot_and_wall(self) -> torch.Tensor:
        return self._render_dot_at(self.dot_position)

    def coord_to_pixel(
        self, locations: torch.Tensor, wall_x=None, door_y=None
    ) -> torch.Tensor:
        """Render images with maze walls and dots at the specified locations.

        Args:
            locations: (bs, t, 2) pixel-space positions
            wall_x, door_y: ignored (each maze layout is unique; we use the
                current episode's maze grid for all batch entries).

        Returns: (bs, t, 2, img_size, img_size)
        """
        if not isinstance(locations, torch.Tensor):
            locations = torch.tensor(
                locations, device=self.device, dtype=torch.float32
            )
        bs, t, _ = locations.shape
        dot_imgs = render_dot(
            locations, self.img_size, self.agent_std, device=self.device
        )  # (bs, t, H, W)
        wall_img = self.wall_img.unsqueeze(0).unsqueeze(0).expand(bs, t, -1, -1)
        return torch.stack([dot_imgs, wall_img], dim=2)  # (bs, t, 2, H, W)
