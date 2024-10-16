from typing import Tuple, Union
import os
import numpy as np
import datetime
import shutil
import copy

import torch
from torch import device

from d4rl import offline_env
from d4rl.kitchen import kitchen_envs
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

import wandb
from gym.core import Wrapper
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from pcdt.data.sequence import GOAL_DIMS

Device = Union[device, str, int, None]


def get_goal(env: offline_env.OfflineEnv, goal_type: str = 'state', goal_frac=None, obs=None):
    if goal_type == 'state' or goal_type == 'state_goal':
        if 'antmaze' in env.spec.id:
            goal = env.target_goal

        elif 'maze2d' in env.spec.id:
            goal = env.goal_locations[0]

        elif 'kitchen' in env.spec.id:
            goal = obs[:30].copy()
            subtask_collection = env.TASK_ELEMENTS
            for task in subtask_collection:
                subtask_indices = kitchen_envs.OBS_ELEMENT_INDICES[task]
                subtask_goals = kitchen_envs.OBS_ELEMENT_GOALS[task]
                goal[subtask_indices] = subtask_goals
            goal_mask = np.ones(30, dtype=np.bool8)
            goal_mask[GOAL_DIMS['kitchen']] = False
            goal = np.where(goal_mask, 0., goal)
        else:
            raise NotImplementedError
    else:  # rtg as goal
        max_score = env.ref_max_score / env._max_episode_steps
        min_score = env.ref_min_score / env._max_episode_steps
        goal = min_score + (max_score - min_score) * goal_frac
        goal = np.array([goal], dtype=np.float32)
    
    return goal


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            dtype = torch.uint8
        else:
            dtype = torch.float32
        return torch.tensor(data=x, dtype=dtype, device=device)
    elif isinstance(x, tuple):
        return torch.tensor(data=x, dtype=torch.float32, device=device)
    elif x is None:
        return None
    else:
        raise NotImplementedError



@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.

    if n_cols is None:
        if v.shape[0] <= 4:
            n_cols = 2
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def create_wandb_video(tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t
    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # tensor: (t, h, w, c)
    tensor = tensor.transpose(0, 3, 1, 2)
    return wandb.Video(tensor, fps=fps, format='mp4')


def record_video(renders=None, n_cols=None, skip_frames=1):
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate([render, np.zeros((max_length - render.shape[0], *render.shape[1:]), dtype=render.dtype)], axis=0)
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    return create_wandb_video(renders, n_cols=n_cols)


def make_ckpt_filename(cfg: DictConfig):
    env_name = cfg.env_name[:-3]
    ckptfile_name = f'{env_name}-s{cfg.seed}'
    if hasattr(cfg, 'info') and cfg.info != '':
        ckptfile_name += f'-{cfg.info}'
    return ckptfile_name


def setup_wandb(cfg: DictConfig, log_dir, dataloader, goal_dim, name=None):
    # Attempt to preprocess all config
    _cfg = copy.deepcopy(cfg)
    _cfg.exp.obs_dim = dataloader.get_obs_dim()
    _cfg.exp.act_dim = dataloader.get_action_dim()
    _cfg.exp.goal_dim = goal_dim
    _cfg = OmegaConf.to_container(_cfg, resolve=True)

    # unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_model = False
    tags = [cfg.env_name] if cfg.wandb.tags is None else cfg.wandb.tags
    wandb_cfg = {
        'config': _cfg,
        'name': name if name is not None else make_ckpt_filename(cfg), 'project': cfg.wandb.project,
        'group': cfg.wandb.group, 'tags': tags,
        'save_dir': log_dir, 'offline': cfg.wandb.offline, 'entity': cfg.wandb.entity,
        'save_code': True,
        'log_model': log_model
    }
    return wandb_cfg


def backup_source(log_dir):
    source_code_backup = os.path.join(log_dir, 'source_codes')
    os.makedirs(log_dir, exist_ok=True)
    shutil.copytree("pcdt", os.path.join(source_code_backup, "pcdt"),
                    ignore=shutil.ignore_patterns('*.pyc', '*.so', '*.cpp', '*.h', '*.pyi', '*.pxd',
                                                  '*.typed', '__pycache__', '*.pkl', '*.npz', '*.pt', '*.ckpt'))


def setup_view_for_env(env, env_name, render_size):
    env.render(mode='rgb_array', width=render_size, height=render_size)
    if 'antmaze-umaze' in env_name:
        env.viewer.cam.lookat[0] = 4
        env.viewer.cam.lookat[1] = 4
        env.viewer.cam.distance = 25
        env.viewer.cam.elevation = -90
    elif 'antmaze-medium' in env_name:
        env.viewer.cam.lookat[0] = 10
        env.viewer.cam.lookat[1] = 10
        env.viewer.cam.distance = 40
        env.viewer.cam.elevation = -90
    elif 'antmaze-large' in env_name:
        env.viewer.cam.lookat[0] = 18
        env.viewer.cam.lookat[1] = 12
        env.viewer.cam.distance = 60
        env.viewer.cam.elevation = -90
    else:
        raise NotImplementedError


class AntMazeVisualizer(Wrapper):
    def __init__(self, env, env_name=None):
        super(AntMazeVisualizer, self).__init__(env)
        n_rows, n_cols = self.env.wrapped_env._np_maze_map.shape
        self.min_x, self.min_y = env.wrapped_env._rowcol_to_xy([0, 0])
        self.max_x, self.max_y = env.wrapped_env._rowcol_to_xy([n_rows - 1, n_cols - 1])

        if 'antmaze-umaze' in env_name or 'antmaze-medium' in env_name:
            correct = 2.5
            self.min_x -= correct
            self.max_x += correct
            self.min_y -= correct
            self.max_y += correct
            self.rectangle_size = 5
        elif 'antmaze-large' in env_name:
            correct = 6
            self.min_x -= 2.7
            self.max_x += 2.7
            self.min_y -= 9
            self.max_y += 9
            self.rectangle_size = 4
        else:
            raise NotImplementedError

    def render(self, mode="human", width=300, height=300, with_goal=False, sub_goal=None, **kwargs):
        if mode == 'rgb_array':
            image = self.env.render(mode, width=width, height=height, **kwargs).transpose(2, 0, 1).copy()
            if with_goal:
                assert width == height, f"width != height ({width}, {height})"
                image = np.rot90(image, k=3, axes=(1, 2))
                goal = self.env.target_goal
                pixx = int((goal[0] - self.min_x) / (self.max_x - self.min_x) * width)
                pixy = int((goal[1] - self.min_y) / (self.max_y - self.min_y) * width)
                image[0, int(pixx - self.rectangle_size):int(pixx + self.rectangle_size), int(pixy - self.rectangle_size):int(pixy + self.rectangle_size)] = 255
                image[1:3, int(pixx - self.rectangle_size):int(pixx + self.rectangle_size), int(pixy - self.rectangle_size):int(pixy + self.rectangle_size)] = 0
                if sub_goal is not None:
                    assert len(sub_goal) == 2
                    pixx = int((sub_goal[0] - self.min_x) / (self.max_x - self.min_x) * width)
                    pixy = int((sub_goal[1] - self.min_y) / (self.max_y - self.min_y) * width)
                    image[0, int(pixx - self.rectangle_size):int(pixx + self.rectangle_size), int(pixy - self.rectangle_size):int(pixy + self.rectangle_size)] = 0
                    image[1, int(pixx - self.rectangle_size):int(pixx + self.rectangle_size), int(pixy - self.rectangle_size):int(pixy + self.rectangle_size)] = 255
                    image[2, int(pixx - self.rectangle_size):int(pixx + self.rectangle_size), int(pixy - self.rectangle_size):int(pixy + self.rectangle_size)] = 0
                image = np.rot90(image, k=1, axes=(1, 2))
            return image
        else:
            return self.env.render(mode, **kwargs)

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps
    

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = np.arange(pos, dtype=np.float32)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

