"""GPU-side maze generator.

DFS maze generation and A* solving are branchy stack-based algorithms that
don't vectorise well on GPU. We keep them on CPU (numpy, ~0.3 ms per 21×21
maze) and move the heavy bit — rendering ``B × T × H × W`` frames — to GPU.

Public surface:
  - ``GPUMazeGenerator.generate_chunk(chunk_size)`` returns a dict matching
    the format produced by ``two_rooms/gpu_precomputed.py`` (so the same
    ``PipelineLoader`` consumes it).
  - ``GPUMazePipelineManager`` mirrors ``GPUPipelineManager`` from the
    two_rooms path: triple-buffered VRAM chunks with a dedicated gen stream.
"""

import numpy as np
import torch

from eb_jepa.datasets.maze.maze_dataset import (
    MazeDatasetConfig,
    cell_to_pixel,
    generate_path_and_actions,
    render_dot,
    render_wall_mask,
)
from eb_jepa.datasets.maze.normalizer import MazeNormalizer
from eb_jepa.datasets.precomputed import PipelineLoader
from eb_jepa.logging import get_logger

logger = get_logger(__name__)


class GPUMazeGenerator:
    """Generate (B, 2, sample_length, H, W) chunks with GPU rendering."""

    def __init__(self, config: MazeDatasetConfig, device, dtype, gen_batch_size=None):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen_batch_size = gen_batch_size
        self.normalizer = MazeNormalizer(img_size=config.img_size)
        self._rng = np.random.default_rng()

        # Cached render grid (float32) for the agent dot.
        img = config.img_size
        lin = torch.arange(img, device=self.device, dtype=torch.float32)
        rr, cc = torch.meshgrid(lin, lin, indexing="ij")
        self._dot_grid = torch.stack([rr, cc], dim=-1)  # (H, W, 2)

    # ------------------------------------------------------------------
    # Sampling — CPU (numpy) since DFS/A* are sequential.
    # ------------------------------------------------------------------

    def _sample_batch_cpu(self, bs):
        cfg = self.config
        n_steps = cfg.n_steps
        n_act = n_steps - 1
        H, W = cfg.maze_height, cfg.maze_width

        mazes = np.empty((bs, H, W), dtype=np.uint8)
        cell_positions = np.empty((bs, n_steps, 2), dtype=np.int32)
        action_vecs = np.zeros((bs, n_act, 2), dtype=np.float32)

        for i in range(bs):
            m, cp, av, _, _ = generate_path_and_actions(cfg, rng=self._rng)
            mazes[i] = m
            cell_positions[i] = cp
            action_vecs[i] = av

        pixel_positions = cell_to_pixel(cell_positions, cfg.cell_size)  # (B, T, 2)
        return mazes, pixel_positions, action_vecs

    # ------------------------------------------------------------------
    # Rendering — vectorised on GPU.
    # ------------------------------------------------------------------

    def _render_walls_gpu(self, mazes_t):
        """mazes_t: (B, H_cell, W_cell) int → (B, img, img) uint8 wall mask."""
        return render_wall_mask(mazes_t, self.config.cell_size)

    def _render_agent_gpu(self, positions):
        """positions: (B, T, 2) float on device → (B, T, img, img) uint8."""
        cfg = self.config
        grid = self._dot_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 2)
        pos = positions.unsqueeze(-2).unsqueeze(-2)       # (B, T, 1, 1, 2)
        d2 = (grid - pos).pow(2).sum(dim=-1)              # (B, T, H, W)
        img = torch.exp(-d2 / (2.0 * cfg.agent_std * cfg.agent_std)) * 255.0
        return img.clamp(0, 255).to(torch.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _generate_batch(self, bs):
        cfg = self.config

        # CPU: maze + path + action vectors
        mazes_np, pixel_positions_np, action_vecs_np = self._sample_batch_cpu(bs)

        # Upload
        mazes_t = torch.from_numpy(mazes_np.astype(np.int64)).to(
            self.device, non_blocking=True
        )                                                                # (B, H, W)
        positions = torch.from_numpy(pixel_positions_np).to(
            self.device, non_blocking=True
        )                                                                # (B, T, 2)
        actions = torch.from_numpy(action_vecs_np).to(
            self.device, non_blocking=True
        )                                                                # (B, T-1, 2)

        # Drop last frame so positions align with actions
        positions = positions[:, :-1]  # (B, T-1, 2)

        # Render walls (static per sample) and agent (varies per timestep)
        walls = self._render_walls_gpu(mazes_t)                # (B, img, img)
        T = positions.shape[1]
        walls = walls.unsqueeze(1).expand(-1, T, -1, -1)        # (B, T, img, img)
        agent = self._render_agent_gpu(positions)               # (B, T, img, img)

        # Stack channels: (B, T, 2, H, W)
        states = torch.stack([agent, walls], dim=2).float()

        if cfg.normalize:
            states = self.normalizer.normalize_state(states)
            positions = self.normalizer.normalize_location(positions)

        # Window sample (random per-batch start)
        sl = cfg.sample_length
        n_act = cfg.n_steps - 1
        max_start = n_act - sl
        starts = torch.randint(0, max(1, max_start + 1), (bs,), device=self.device)
        tidx = starts[:, None] + torch.arange(sl, device=self.device)[None, :]
        b_ix = torch.arange(bs, device=self.device)[:, None]

        states_w = states[b_ix, tidx].permute(0, 2, 1, 3, 4)   # (B, 2, sl, H, W)
        actions_w = actions[b_ix, tidx].permute(0, 2, 1)       # (B, 2, sl)
        positions_w = positions[b_ix, tidx].permute(0, 2, 1)   # (B, 2, sl)

        # Dummies for WallSample compat
        wall_x = torch.zeros(bs, device=self.device)
        door_y = torch.zeros(bs, device=self.device)

        return {
            "states": states_w,
            "actions": actions_w,
            "locations": positions_w,
            "wall_x": wall_x,
            "door_y": door_y,
        }

    def generate_chunk(self, chunk_size):
        gb = self.gen_batch_size or chunk_size
        parts = []
        done = 0
        while done < chunk_size:
            b = min(gb, chunk_size - done)
            parts.append(self._generate_batch(b))
            done += b
        if len(parts) == 1:
            chunk = parts[0]
        else:
            chunk = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
        for k in ("states", "actions", "locations"):
            chunk[k] = chunk[k].to(self.dtype)
        return chunk


# ---------------------------------------------------------------------------
# Pipeline manager — triple-buffered, mirrors GPUPipelineManager.
# ---------------------------------------------------------------------------


class GPUMazePipelineManager:
    """Double-buffered VRAM chunks fed by ``GPUMazeGenerator``.

    Same interface as ``two_rooms.GPUPipelineManager`` so the existing
    ``PipelineLoader`` consumes it unchanged.
    """

    def __init__(self, config: MazeDatasetConfig, chunk_size, device, dtype, gen_batch_size=None):
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = GPUMazeGenerator(
            config, device=self.device, dtype=dtype, gen_batch_size=gen_batch_size
        )
        self.gen_stream = torch.cuda.Stream(device=self.device)

        self.current = None
        self.next = None
        self._pending = None

    def warm_up(self):
        with torch.cuda.stream(self.gen_stream):
            self.current = self.generator.generate_chunk(self.chunk_size)
            self.next = self.generator.generate_chunk(self.chunk_size)
        self.gen_stream.synchronize()
        with torch.cuda.stream(self.gen_stream):
            self._pending = self.generator.generate_chunk(self.chunk_size)

    def swap(self):
        self.gen_stream.synchronize()
        default = torch.cuda.current_stream(self.device)
        if self.current is not None:
            for v in self.current.values():
                v.record_stream(default)
        self.current = self.next
        self.next = self._pending
        with torch.cuda.stream(self.gen_stream):
            self._pending = self.generator.generate_chunk(self.chunk_size)

    def shutdown(self):
        self.gen_stream.synchronize()
        self.current = self.next = self._pending = None


def init_gpu_maze_data(
    config: MazeDatasetConfig,
    chunk_size,
    epoch_size,
    batch_size,
    device,
    dtype,
    gen_batch_size=None,
    drop_last=True,
):
    manager = GPUMazePipelineManager(
        config=config,
        chunk_size=chunk_size,
        device=device,
        dtype=dtype,
        gen_batch_size=gen_batch_size,
    )
    loader = PipelineLoader(
        manager=manager,
        batch_size=batch_size,
        epoch_size=epoch_size,
        drop_last=drop_last,
        normalizer=MazeNormalizer(img_size=config.img_size),
    )
    return loader, manager
