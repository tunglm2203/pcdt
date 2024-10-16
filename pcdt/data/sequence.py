from typing import Dict, Optional, TypeVar
from collections import namedtuple

import d4rl
import gym

import os
import pickle
import random
import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, default_convert, default_collate
from omegaconf import DictConfig

from .parse_d4rl import parse_pickle_datasets, DATASET_DIR


GOAL_DIMS = {
    'maze': [0, 1],
    'kitchen': [11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
}

T = TypeVar('T')
Batch = namedtuple('Batch', ['observations', 'actions', 'goal', 'valid_length', 'padding_mask'])
Batch_v1 = namedtuple('Batch_v1', ['observations', 'actions', 'goal', 'valid_length', 'padding_mask', 'timesteps'])


def obs_to_goal(env_name, obs):
    if 'maze' in env_name:  # antmaze provides goal as (x,y), the first two dimensions in the state space
        goal = obs[GOAL_DIMS['maze']]
    elif 'kitchen' in env_name:  # zero out redundant dimensions such as robot proprioceptive state
        goal_mask = np.ones(30, dtype=np.bool_)
        goal_mask[GOAL_DIMS['kitchen']] = False
        goal = np.where(goal_mask, 0., obs)  # already removed the last 30 dimension during preprocessing
    return goal


def preprocess_d4rl_episodes_from_path(path: str, max_episode_length: int, number: Optional[int] = None, proportion: Optional[float] = None):
    '''
    read dataset from pickle file
    '''
    with open(path, 'rb') as f:
        episodes = pickle.load(f)

    n_episode = len(episodes)
    
    print(f'Loading dataset from {path}: {n_episode} episodes')

    episode_lengths = [e['rewards'].shape[0] for e in episodes]
    key_dims = {key: episodes[0][key].shape[-1] for key in episodes[0].keys()}
    buffers = {key: np.zeros((n_episode, max_episode_length, key_dims[key]), dtype=np.float32) for key in episodes[0].keys()}

    for idx, e in enumerate(episodes):  # put episodes into fix sized numpy arrays (for padding)
        for key, buffer in buffers.items():
            buffer[idx, :episode_lengths[idx]] = e[key]
    
    buffers['episode_lengths'] = np.array(episode_lengths)

    return buffers


def fix_terminals_antmaze(dataset):
    # Preprocess dataset follows the strategy from HIQL's paper
    dones_float = np.zeros_like(dataset['rewards'])
    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1
    return dones_float


class SequenceDataset(Dataset):
    def __init__(self, env_name: str, epi_buffers: Dict, max_episode_length: int = 1000, ctx_size: int = 100, goal_type: str = 'state'):
        super().__init__()
        self.env_name = env_name
        self.max_episode_length = max_episode_length  # max length for each episode (env setting)
        self.ctx_size = ctx_size  # context window size
        self.goal_type = goal_type
        
        self.epi_buffers = epi_buffers
        self.traj_indices = self.sample_trajs_from_episode()

    def sample_trajs_from_episode(self):
        '''
            makes indices for sampling from dataset;
            each index maps to a trajectory (start, end)
        '''
        indices = []
        for i, epi_length in enumerate(self.epi_buffers['episode_lengths']):
            max_start = epi_length - self.ctx_size
            for start in range(max_start):
                end = start + self.ctx_size
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.traj_indices)

    def __getitem__(self, idx: int):
        epi_idx, start, end = self.traj_indices[idx]

        epi_length = self.epi_buffers['episode_lengths'][epi_idx]
        observations = self.epi_buffers['observations'][epi_idx, start:end]
        actions = self.epi_buffers['actions'][epi_idx, start:end]

        valid_length = min(epi_length, end) - start

        if self.goal_type == 'state':
            goal_idx = random.choice(range(end, epi_length))   # randomly sample a goal from the sequence after context window
            goal = self.epi_buffers['observations'][epi_idx, goal_idx]
            goal = obs_to_goal(self.env_name, goal)
            goal = np.expand_dims(goal, axis=0)
        elif self.goal_type == 'state_goal':
            goal_idx = random.choice(range(end, epi_length))  # randomly sample a goal from the sequence after context window
            goal = self.epi_buffers['observations'][epi_idx, goal_idx]
            if 'kitchen' in self.env_name:
                goal = obs_to_goal(self.env_name, goal)
            goal = np.expand_dims(goal, axis=0)
        else:
            goal = self.epi_buffers['avg_rtgs'][epi_idx, start:end]

        padding_mask = np.zeros(shape=(self.ctx_size, ), dtype=np.bool_)  # only consider one modality length
        padding_mask[valid_length:] = True

        batch = Batch(observations, actions, goal, valid_length, padding_mask)
        return batch


# NOTE: Returned batch includes 'timestep'
class SequenceDataset_v1(Dataset):
    def __init__(self, env_name: str, epi_buffers: Dict, max_episode_length: int = 1000, ctx_size: int = 100,
                 goal_type: str = 'state'):
        super().__init__()
        self.env_name = env_name
        self.max_episode_length = max_episode_length  # max length for each episode (env setting)
        self.ctx_size = ctx_size  # context window size
        self.goal_type = goal_type

        self.epi_buffers = epi_buffers
        self.traj_indices = self.sample_trajs_from_episode()

    def sample_trajs_from_episode(self):
        '''
            makes indices for sampling from dataset;
            each index maps to a trajectory (start, end)
        '''
        indices = []
        for i, epi_length in enumerate(self.epi_buffers['episode_lengths']):
            max_start = epi_length - self.ctx_size
            for start in range(max_start):
                end = start + self.ctx_size
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.traj_indices)

    def __getitem__(self, idx: int):
        epi_idx, start, end = self.traj_indices[idx]

        epi_length = self.epi_buffers['episode_lengths'][epi_idx]
        observations = self.epi_buffers['observations'][epi_idx, start:end]
        actions = self.epi_buffers['actions'][epi_idx, start:end]
        timesteps = self.epi_buffers['timesteps'][epi_idx, start:end]

        valid_length = min(epi_length, end) - start

        if self.goal_type == 'state':
            goal_idx = random.choice(range(end, epi_length))  # randomly sample a goal from the sequence after context window
            goal = self.epi_buffers['observations'][epi_idx, goal_idx]
            goal = obs_to_goal(self.env_name, goal)
            goal = np.expand_dims(goal, axis=0)
        elif self.goal_type == 'state_goal':
            goal_idx = random.choice(range(end, epi_length))  # randomly sample a goal from the sequence after context window
            goal = self.epi_buffers['observations'][epi_idx, goal_idx]
            if 'kitchen' in self.env_name:
                goal = obs_to_goal(self.env_name, goal)
            goal = np.expand_dims(goal, axis=0)
        else:
            goal = self.epi_buffers['avg_rtgs'][epi_idx, start:end]

        padding_mask = np.zeros(shape=(self.ctx_size,), dtype=np.bool_)  # only consider one modality length
        padding_mask[valid_length:] = True
        batch = Batch_v1(observations, actions, goal, valid_length, padding_mask, timesteps)
        return batch


class PretrainDataset(SequenceDataset):
    def sample_trajs_from_episode(self):
        '''
            makes start indices for sampling from dataset
        '''
        indices = []
        for i, epi_length in enumerate(self.epi_buffers['episode_lengths']):
            max_start = epi_length - self.ctx_size
            for start in range(max_start):
                indices.append((i, start))
        indices = np.array(indices)
        return indices

    def __getitem__(self, idx: int):
        epi_idx, start = self.traj_indices[idx]
        epi_length = self.epi_buffers['episode_lengths'][epi_idx]

        end = random.choice(range(start + self.ctx_size, epi_length))
        valid_length = end - start

        observations = self.epi_buffers['observations'][epi_idx, start:end]
        actions = self.epi_buffers['actions'][epi_idx, start:end]

        if self.goal_type == 'state':
            goal = self.epi_buffers['observations'][epi_idx, end]
            goal = obs_to_goal(self.env_name, goal)
            goal = np.expand_dims(goal, axis=0)
        elif self.goal_type == 'state_goal':
            goal = self.epi_buffers['observations'][epi_idx, end]
            if 'kitchen' in self.env_name:
                goal = obs_to_goal(self.env_name, goal)
            goal = np.expand_dims(goal, axis=0)
        else:
            goal = self.epi_buffers['avg_rtgs'][epi_idx, start:start + self.ctx_size]

        return observations, actions, goal, valid_length


def pt_collate_fn(batch):
    batch_size = len(batch)
    obss = default_convert([item[0] for item in batch])
    acts = default_convert([item[1] for item in batch])
    goal = default_collate([item[2] for item in batch])
    valid_lengths = [item[3] for item in batch]

    max_valid_length = max(valid_lengths)
    pad_observations = torch.zeros(batch_size, max_valid_length, obss[0].shape[-1])
    pad_actions = torch.zeros(batch_size, max_valid_length, acts[0].shape[-1])
    padding_mask = torch.zeros(batch_size, max_valid_length, dtype=torch.bool)
    for idx, item in enumerate(zip(obss, acts, valid_lengths)):
        obs, act, valid_len = item
        pad_observations[idx, :valid_len] = obs
        pad_actions[idx, :valid_len] = act
        padding_mask[idx, valid_len:] = True
    valid_lengths = torch.tensor(valid_lengths)
    batch = Batch(pad_observations, pad_actions, goal, valid_lengths, padding_mask)
    return batch


class D4RLDataModule(pl.LightningDataModule):
    def __init__(self, goal_type, config: DictConfig):
        super().__init__()
        self.env_name = config.env_name
        self.batch_size = config.exp.batch_size
        self.ctx_size = config.exp.ctx_size
        self.max_episode_length = config.max_episode_length
        self.num_workers = config.num_workers
        self.train_size = config.train_size
        self.goal_type = goal_type
        self.dataset_path = os.path.join(DATASET_DIR, f'{self.env_name}.pkl')

    # def prepare_data(self):  # may comment out if not needed
    #     parse_pickle_datasets(self.env_name, self.dataset_path)

    def setup(self, stage: str):
        epi_buffer = preprocess_d4rl_episodes_from_path(self.dataset_path, self.max_episode_length)
        # split train/val and put into sequencedataset or goaldataset
        self._obs_dim = epi_buffer['observations'].shape[-1]
        self._action_dim = epi_buffer['actions'].shape[-1]

        num_epis = epi_buffer['rewards'].shape[0]
        if self.train_size <= 1:
            num_train = int(num_epis * self.train_size)
        else:
            num_train = self.train_size
        indices = np.arange(num_epis)

        train_indices = indices[:num_train]
        val_indices = indices[num_train: num_epis]
        train_buffer = {key: value[train_indices] for key, value in epi_buffer.items()}
        val_buffer = {key: value[val_indices] for key, value in epi_buffer.items()}

        if stage == 'trl':  # trajectory representation learning
            self.train = PretrainDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = PretrainDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
        else:
            self.train = SequenceDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = SequenceDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)


    def train_dataloader(self):
        if isinstance(self.train, PretrainDataset):
            return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if isinstance(self.val, PretrainDataset):
            return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def get_obs_dim(self):
        return self._obs_dim
    
    def get_action_dim(self):
        return self._action_dim


# NOTE: This treats goal as final state, only use for PCDT
class D4RLDataModule_StateGoal_v0(pl.LightningDataModule):
    def __init__(self, goal_type, config: DictConfig, clip_to_eps: bool = True, eps: float = 1e-5,):
        super().__init__()
        self.env_name = config.env_name
        self.batch_size = config.exp.batch_size
        self.ctx_size = config.exp.ctx_size
        self.max_episode_length = config.max_episode_length
        self.num_workers = config.num_workers
        self.train_size = config.train_size
        self.goal_type = goal_type
        self.dataset_path = os.path.join(DATASET_DIR, f'{self.env_name}.pkl')

        self.clip_to_eps = clip_to_eps
        self.eps = eps

    # def prepare_data(self):  # may comment out if not needed
    #     parse_pickle_datasets(self.env_name, self.dataset_path)

    def setup(self, stage: str):
        env = gym.make(self.env_name)
        dataset = d4rl.qlearning_dataset(env)

        print(f"Processing data ...")
        if self.clip_to_eps:
            lim = 1 - self.eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        # Preprocess dataset follows the strategy from HIQL's paper
        if 'antmaze' in self.env_name:
            dataset['terminals'][:] = 0.
            dones_float = fix_terminals_antmaze(dataset)
            dataset['terminals'] = dones_float

            # Convert the dataset into format 'key': NxT, N=999, T=1000
            terminal_locs, = np.nonzero(dataset['terminals'] > 0)
            if 'antmaze-umaze' in self.env_name:
                terminal_locs = terminal_locs[:-1]

            if 'umaze-v2' in self.env_name:
                self.max_episode_length = 700  # Fix this for only antmaze-umaze-v2

            n_episodes = len(terminal_locs)
            epi_buffer = {
                'actions': np.empty((n_episodes, self.max_episode_length, dataset['actions'].shape[-1]), dtype=np.float32),
                'observations': np.empty((n_episodes, self.max_episode_length, dataset['observations'].shape[-1]), dtype=np.float32),
                'rewards': np.empty((n_episodes, self.max_episode_length, 1), dtype=np.float32),
                'terminals': np.zeros((n_episodes, self.max_episode_length, 1)),
                'episode_lengths': np.ones((n_episodes,), dtype=np.int64) * self.max_episode_length,
                # 'timesteps': np.empty((n_episodes, self.max_episode_length), dtype=np.float32)
            }

            ep_length = self.max_episode_length
            for i, terminal_idx in enumerate(terminal_locs.tolist()):
                epi_buffer['observations'][i] = dataset['observations'][i * ep_length: terminal_idx + 1]
                epi_buffer['actions'][i] = dataset['actions'][i * ep_length: terminal_idx + 1]
                epi_buffer['rewards'][i] = dataset['rewards'][i * ep_length: terminal_idx + 1][:, None]
                epi_buffer['terminals'][i] = dataset['terminals'][i * ep_length: terminal_idx + 1][:, None]
                # epi_buffer['timesteps'][i] = np.arange(0, terminal_idx - i * ep_length + 1)
        elif 'kitchen' in self.env_name:
            n_dim_obs = 30  # remove the originally labeled goal from kitchen's observation space (same with PCDT, HIQL)

            # NOTE: filter terminals following HIQL
            non_last_idx = np.nonzero(~dataset['terminals'])[0]
            last_idx = np.nonzero(dataset['terminals'])[0]
            penult_idx = last_idx - 1
            new_dataset = dict()
            for k, v in dataset.items():
                if k == 'terminals':
                    v[penult_idx] = 1
                new_dataset[k] = v[non_last_idx]
            dataset = new_dataset

            terminal_locs, = np.nonzero(dataset['terminals'] > 0)
            n_episodes = len(terminal_locs)
            epi_buffer = {
                'observations': np.zeros((n_episodes, self.max_episode_length, n_dim_obs), dtype=np.float32),
                'actions': np.zeros((n_episodes, self.max_episode_length, dataset['actions'].shape[-1]), dtype=np.float32),
                'rewards': np.zeros((n_episodes, self.max_episode_length, 1), dtype=np.float32),
                'terminals': np.zeros((n_episodes, self.max_episode_length, 1)),
                'episode_lengths': np.zeros((n_episodes,), dtype=np.int64),
                # 'timesteps': np.zeros((n_episodes, self.max_episode_length), dtype=np.float32)
            }

            last_terminal_idx = 0
            for i, terminal_idx in enumerate(terminal_locs.tolist()):
                epi_buffer['observations'][i][:terminal_locs[i] + 1 - last_terminal_idx] = dataset['observations'][last_terminal_idx: terminal_locs[i] + 1][:, :n_dim_obs]
                epi_buffer['actions'][i][:terminal_locs[i] + 1 - last_terminal_idx] = dataset['actions'][last_terminal_idx: terminal_locs[i] + 1]
                epi_buffer['rewards'][i][:terminal_locs[i] + 1 - last_terminal_idx] = dataset['rewards'][last_terminal_idx: terminal_locs[i] + 1][:, None]
                epi_buffer['terminals'][i][:terminal_locs[i] + 1 - last_terminal_idx] = dataset['terminals'][last_terminal_idx: terminal_locs[i] + 1][:, None]
                # epi_buffer['timesteps'][i][:terminal_locs[i] + 1 - last_terminal_idx] = np.arange(0, terminal_locs[i] - last_terminal_idx + 1)
                epi_buffer['episode_lengths'][i] = terminal_locs[i] - last_terminal_idx + 1
                last_terminal_idx = terminal_locs[i] + 1
        else:
            raise NotImplementedError

        print(f"Processing completed.\n")

        # split train/val and put into sequencedataset or goaldataset
        self._obs_dim = epi_buffer['observations'].shape[-1]
        self._action_dim = epi_buffer['actions'].shape[-1]

        num_epis = epi_buffer['rewards'].shape[0]
        if self.train_size <= 1:
            num_train = int(num_epis * self.train_size)
        else:
            num_train = self.train_size
        indices = np.arange(num_epis) 

        train_indices = indices[:num_train]
        val_indices = indices[num_train: num_epis]
        train_buffer = {key: value[train_indices] for key, value in epi_buffer.items()}
        val_buffer = {key: value[val_indices] for key, value in epi_buffer.items()}

        if stage == 'trl':  # trajectory representation learning
            self.train = PretrainDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = PretrainDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
        else:
            self.train = SequenceDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = SequenceDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)

    def train_dataloader(self):
        if isinstance(self.train, PretrainDataset):
            return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if isinstance(self.val, PretrainDataset):
            return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def get_obs_dim(self):
        return self._obs_dim

    def get_action_dim(self):
        return self._action_dim


# NOTE: This treats goal as final state, return batch contains timestep (only in policy learning), used for DT
class D4RLDataModule_StateGoal_v1(pl.LightningDataModule):
    def __init__(self, goal_type, config: DictConfig, clip_to_eps: bool = True, eps: float = 1e-5,):
        super().__init__()
        self.env_name = config.env_name
        self.batch_size = config.exp.batch_size
        self.ctx_size = config.exp.ctx_size
        self.max_episode_length = config.max_episode_length
        self.num_workers = config.num_workers
        self.train_size = config.train_size
        self.goal_type = goal_type
        self.dataset_path = os.path.join(DATASET_DIR, f'{self.env_name}.pkl')

        self.clip_to_eps = clip_to_eps
        self.eps = eps

    # def prepare_data(self):  # may comment out if not needed
    #     parse_pickle_datasets(self.env_name, self.dataset_path)

    def setup(self, stage: str):
        env = gym.make(self.env_name)
        dataset = d4rl.qlearning_dataset(env)

        print(f"Processing data ...")
        if self.clip_to_eps:
            lim = 1 - self.eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        # Preprocess dataset follows the strategy from HIQL's paper
        if 'antmaze' in self.env_name:
            dataset['terminals'][:] = 0.
            dones_float = fix_terminals_antmaze(dataset)
            dataset['terminals'] = dones_float

            # Convert the dataset into format 'key': NxT, N=999, T=1000
            terminal_locs, = np.nonzero(dataset['terminals'] > 0)
            if 'antmaze-umaze' in self.env_name:
                terminal_locs = terminal_locs[:-1]

            if 'umaze-v2' in self.env_name:
                self.max_episode_length = 700  # Fix this for only antmaze-umaze-v2

            n_episodes = len(terminal_locs)
            epi_buffer = {
                'actions': np.empty((n_episodes, self.max_episode_length, dataset['actions'].shape[-1]), dtype=np.float32),
                'observations': np.empty((n_episodes, self.max_episode_length, dataset['observations'].shape[-1]), dtype=np.float32),
                'rewards': np.empty((n_episodes, self.max_episode_length, 1), dtype=np.float32),
                'terminals': np.zeros((n_episodes, self.max_episode_length, 1)),
                'episode_lengths': np.ones((n_episodes,), dtype=np.int64) * self.max_episode_length,
                'timesteps':np.empty((n_episodes, self.max_episode_length), dtype=np.float32)
            }

            ep_length = self.max_episode_length
            for i, terminal_idx in enumerate(terminal_locs.tolist()):
                epi_buffer['observations'][i] = dataset['observations'][i * ep_length: terminal_idx + 1]
                epi_buffer['actions'][i] = dataset['actions'][i * ep_length: terminal_idx + 1]
                epi_buffer['rewards'][i] = dataset['rewards'][i * ep_length: terminal_idx + 1][:, None]
                epi_buffer['terminals'][i] = dataset['terminals'][i * ep_length: terminal_idx + 1][:, None]
                epi_buffer['timesteps'][i]= np.arange(0, terminal_idx - i * ep_length + 1)
        elif 'kitchen' in self.env_name:
            n_dim_obs = 30  # remove the originally labeled goal from kitchen's observation space (same with PCDT, HIQL)

            # NOTE: filter terminals following HIQL
            non_last_idx = np.nonzero(~dataset['terminals'])[0]
            last_idx = np.nonzero(dataset['terminals'])[0]
            penult_idx = last_idx - 1
            new_dataset = dict()
            for k, v in dataset.items():
                if k == 'terminals':
                    v[penult_idx] = 1
                new_dataset[k] = v[non_last_idx]
            dataset = new_dataset

            terminal_locs, = np.nonzero(dataset['terminals'] > 0)
            n_episodes = len(terminal_locs)
            epi_buffer = {
                'observations': np.zeros((n_episodes, self.max_episode_length, n_dim_obs), dtype=np.float32),
                'actions': np.zeros((n_episodes, self.max_episode_length, dataset['actions'].shape[-1]), dtype=np.float32),
                'rewards': np.zeros((n_episodes, self.max_episode_length, 1), dtype=np.float32),
                'terminals': np.zeros((n_episodes, self.max_episode_length, 1)),
                'episode_lengths': np.zeros((n_episodes,), dtype=np.int64),
                'timesteps': np.zeros((n_episodes, self.max_episode_length), dtype=np.float32)
            }

            last_terminal_idx = 0
            for i, terminal_idx in enumerate(terminal_locs.tolist()):
                epi_buffer['observations'][i][:terminal_locs[i] + 1 - last_terminal_idx] = dataset['observations'][last_terminal_idx: terminal_locs[i] + 1][:, :n_dim_obs]
                epi_buffer['actions'][i][:terminal_locs[i] + 1 - last_terminal_idx] = dataset['actions'][last_terminal_idx: terminal_locs[i] + 1]
                epi_buffer['rewards'][i][:terminal_locs[i] + 1 - last_terminal_idx] = dataset['rewards'][last_terminal_idx: terminal_locs[i] + 1][:, None]
                epi_buffer['terminals'][i][:terminal_locs[i] + 1 - last_terminal_idx] = dataset['terminals'][last_terminal_idx: terminal_locs[i] + 1][:, None]
                epi_buffer['timesteps'][i][:terminal_locs[i] + 1 - last_terminal_idx]= np.arange(0, terminal_locs[i] - last_terminal_idx + 1)
                epi_buffer['episode_lengths'][i] = terminal_locs[i] - last_terminal_idx + 1
                last_terminal_idx = terminal_locs[i] + 1
        else:
            raise NotImplementedError

        print(f"Processing completed.\n")

        # split train/val and put into sequencedataset or goaldataset
        self._obs_dim = epi_buffer['observations'].shape[-1]
        self._action_dim = epi_buffer['actions'].shape[-1]

        num_epis = epi_buffer['rewards'].shape[0]
        if self.train_size <= 1:
            num_train = int(num_epis * self.train_size)
        else:
            num_train = self.train_size
        indices = np.arange(num_epis)

        train_indices = indices[:num_train]
        val_indices = indices[num_train: num_epis]
        train_buffer = {key: value[train_indices] for key, value in epi_buffer.items()}
        val_buffer = {key: value[val_indices] for key, value in epi_buffer.items()}

        if stage == 'trl':  # trajectory representation learning
            self.train = PretrainDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = PretrainDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
        else:
            self.train = SequenceDataset_v1(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = SequenceDataset_v1(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)

    def train_dataloader(self):
        if isinstance(self.train, PretrainDataset):
            return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if isinstance(self.val, PretrainDataset):
            return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def get_obs_dim(self):
        return self._obs_dim

    def get_action_dim(self):
        return self._action_dim


class SawyerDataModule_StateGoal_v0(pl.LightningDataModule):
    def __init__(self, goal_type, config: DictConfig, clip_to_eps: bool = True, eps: float = 1e-5,):
        super().__init__()
        self.env_name = config.env_name
        self.batch_size = config.exp.batch_size
        self.ctx_size = config.exp.ctx_size
        self.max_episode_length = config.max_episode_length
        self.num_workers = config.num_workers
        self.train_size = config.train_size
        self.goal_type = goal_type
        self.dataset_path = os.path.join(DATASET_DIR, f'{self.env_name}.pkl')

        self.clip_to_eps = clip_to_eps
        self.eps = eps

    # def prepare_data(self):  # may comment out if not needed
    #     parse_pickle_datasets(self.env_name, self.dataset_path)

    def setup(self, stage: str):
        path = 'sawyer_dataset/sawyer-reacher-mixed-v0.pkl'
        with open(path, 'rb') as f:
            epi_buffer = pickle.load(f)

        if self.clip_to_eps:
            lim = 1 - self.eps
            epi_buffer['actions'] = np.clip(epi_buffer['actions'], -lim, lim)

        # split train/val and put into sequencedataset or goaldataset
        self._obs_dim = epi_buffer['observations'].shape[-1]
        self._action_dim = epi_buffer['actions'].shape[-1]

        num_epis = epi_buffer['rewards'].shape[0]
        indices = np.arange(num_epis)

        train_indices = indices
        val_indices = indices
        train_buffer = {key: value[train_indices] for key, value in epi_buffer.items()}
        val_buffer = {key: value[val_indices] for key, value in epi_buffer.items()}

        if stage == 'trl':  # trajectory representation learning
            self.train = PretrainDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = PretrainDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
        else:
            self.train = SequenceDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = SequenceDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)

    def train_dataloader(self):
        if isinstance(self.train, PretrainDataset):
            return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if isinstance(self.val, PretrainDataset):
            return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def get_obs_dim(self):
        return self._obs_dim

    def get_action_dim(self):
        return self._action_dim


class SawyerDataModule_StateGoal_v1(pl.LightningDataModule):
    def __init__(self, goal_type, config: DictConfig, clip_to_eps: bool = True, eps: float = 1e-5,):
        super().__init__()
        self.env_name = config.env_name
        self.batch_size = config.exp.batch_size
        self.ctx_size = config.exp.ctx_size
        self.max_episode_length = config.max_episode_length
        self.num_workers = config.num_workers
        self.train_size = config.train_size
        self.goal_type = goal_type
        self.dataset_path = os.path.join(DATASET_DIR, f'{self.env_name}.pkl')

        self.clip_to_eps = clip_to_eps
        self.eps = eps

    # def prepare_data(self):  # may comment out if not needed
    #     parse_pickle_datasets(self.env_name, self.dataset_path)

    def setup(self, stage: str):
        path = 'sawyer_dataset/sawyer-reacher-mixed-v0.pkl'
        with open(path, 'rb') as f:
            epi_buffer = pickle.load(f)

        if self.clip_to_eps:
            lim = 1 - self.eps
            epi_buffer['actions'] = np.clip(epi_buffer['actions'], -lim, lim)

        # split train/val and put into sequencedataset or goaldataset
        self._obs_dim = epi_buffer['observations'].shape[-1]
        self._action_dim = epi_buffer['actions'].shape[-1]

        num_epis = epi_buffer['rewards'].shape[0]
        indices = np.arange(num_epis)

        train_indices = indices
        val_indices = indices
        train_buffer = {key: value[train_indices] for key, value in epi_buffer.items()}
        val_buffer = {key: value[val_indices] for key, value in epi_buffer.items()}

        if stage == 'trl':  # trajectory representation learning
            self.train = PretrainDataset(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = PretrainDataset(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
        else:
            self.train = SequenceDataset_v1(self.env_name, train_buffer, self.max_episode_length, self.ctx_size, self.goal_type)
            self.val = SequenceDataset_v1(self.env_name, val_buffer, self.max_episode_length, self.ctx_size, self.goal_type)

    def train_dataloader(self):
        if isinstance(self.train, PretrainDataset):
            return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if isinstance(self.val, PretrainDataset):
            return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def get_obs_dim(self):
        return self._obs_dim

    def get_action_dim(self):
        return self._action_dim