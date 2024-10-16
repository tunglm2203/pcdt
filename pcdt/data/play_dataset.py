import os
import pickle
import torch
import logging

import d4rl  # noqa
import gym

from typing import Optional, TypeVar, Dict, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader, default_convert, default_collate
from collections import namedtuple
import numpy as np
import random
import pytorch_lightning as pl
from typing import Dict
from omegaconf import DictConfig, OmegaConf

import hydra

logger = logging.getLogger(__name__)


class D4RLPlayDataset(Dataset):
    """
    Dataset Loader that uses a shared memory cache

    parameters
    ----------

    data_dir:           path of folder containing episode files
    save_format:        format of episodes in datasets_dir (.pkl or .npz)
    obs_space:          DictConfig of the observation modalities of the dataset
    max_window_size:    maximum length of the episodes sampled from the dataset
    """

    def __init__(
        self,
        episode_buffer: Dict,
        min_window_size: int = 8,
        max_window_size: int = 16,
        pad: bool = True,
        transf_type: str = "train",
        include_goal: bool = False,
        goal_sampling_prob: float = 0.3,
        goal_augmentation: bool = False,
        goal_threshold: float = 0.5,
    ):

        self.dataset = episode_buffer

        self.pad = pad
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

        self.episode_lookup: List[int] = []
        # self.transform_manager = transform_manager
        self.goal_augmentation = goal_augmentation
        self.transf_type = transf_type
        self.episode_lookup, self.max_batched_length_per_demo = self.load_file_indices()
        self.include_goal = include_goal
        self.goal_sampling_prob = goal_sampling_prob
        self.goal_threshold = goal_threshold

    def __len__(self) -> int:
        """
        returns
        ----------
        number of possible starting frames
        """
        return len(self.episode_lookup)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = np.random.randint(
                    self.min_window_size, self.max_window_size + 1
                )
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} "
                    f"> max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        sequence = self.get_sequences(idx, window_size)
        if self.pad:
            sequence = self.pad_sequence(sequence, window_size)

        if self.include_goal:
            sequence["goal"], sequence["goal_reached"] = self.get_future_goal(
                idx, window_size
            )
        return sequence

    def get_file(self, file_idx, transform=True):
        obs = {
            "observations": self.dataset["observations"][file_idx],
            "actions": self.dataset["actions"][file_idx],
        }
        # if transform:
        #     return self.transform_manager(obs, transf_type=self.transf_type)
        return obs

    def find_episode_end(self, step):
        for start_step, end_step in self.ep_start_end_ids:
            if start_step <= step <= end_step:
                return end_step
        return None

    def get_random_state(self):
        file_idx = np.random.choice(self.episode_lookup)
        return self.get_file(file_idx, transform=True)

    def extract_goal_from_state(self, state):
        goal = state["observations"][:2]
        # Augmentation for the goal
        if self.goal_augmentation:
            goal += np.random.uniform(low=-0.1, high=0.1, size=2)
        return goal

    def get_future_goal(self, idx, window_size):
        """Tries to return a random future state between
        [seq_end, episode_end] (Low and high inclusive)
        following the geometric distribution"""

        seq_start = self.episode_lookup[idx]
        episode_end = self.find_episode_end(seq_start)
        if episode_end is None:
            goal_state = self.get_random_state()
            goal = self.extract_goal_from_state(goal_state)
        else:
            disp = np.random.default_rng().geometric(p=self.goal_sampling_prob)
            goal_step = seq_start + (window_size - 1) * disp
            if self.goal_augmentation:
                noise_step = np.random.randint(3) - 1
                goal_step += noise_step
            file_step = min(episode_end, goal_step)
            goal_state = self.get_file(file_step, transform=True)
            goal = self.extract_goal_from_state(goal_state)

        seq_end_pos = self.dataset["observations"][seq_start + window_size - 1][:2]
        reached = np.linalg.norm(goal - seq_end_pos) < self.goal_threshold
        return goal, reached

    def pad_sequence(
        self,
        seq: Dict,
        window_size: int,
    ) -> Dict:

        # Update modalities
        pad_size = self.max_window_size - window_size
        # Zero pad all and repeat the gripper action
        seq["actions"] = self.pad_with_zeros(seq["actions"], pad_size)
        seq["observations"] = self.pad_with_repetition(seq["observations"], pad_size)
        return seq

    @staticmethod
    def pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """repeats the last element with pad_size"""
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """repeats the last element with pad_size"""
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        window_size:    length of sampled episode

        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """

        start_file_indx = self.episode_lookup[idx]
        end_file_indx = start_file_indx + window_size
        actions = self.dataset["actions"][start_file_indx:end_file_indx]
        observations = self.dataset["observations"][start_file_indx:end_file_indx]
        seq = {"actions": default_convert(actions), "observations": default_convert(observations)}
        # Apply transformations
        # seq = self.transform_manager(seq, transf_type=self.transf_type)
        # Add info
        seq["idx"] = idx
        seq["window_size"] = window_size
        return seq

    def set_ep_start_end_ids(self):
        timeouts = self.dataset["timeouts"].nonzero()[0]
        terminals = self.dataset["terminals"].nonzero()[0]
        episode_ends = list(set(timeouts.tolist() + terminals.tolist()))
        episode_ends.sort()

        ep_start_end_ids = []
        start = 0
        for ep_end in episode_ends:
            if ep_end - start > self.min_window_size:
                ep_start_end_ids.append([start, ep_end])
            start = ep_end + 1
        self.ep_start_end_ids = ep_start_end_ids

    def load_file_indices(self) -> Tuple[List, List]:
        """
        this method builds the mapping from index to file_name used
        for loading the episodes

        parameters
        ----------
        abs_datasets_dir: absolute path of the directory containing the dataset

        returns
        ----------
        episode_lookup: list for the mapping from training example index
                        to episode (file) index
        max_batched_length_per_demo: list of possible starting indices per episode
        """
        self.set_ep_start_end_ids()

        episode_lookup = []
        max_batched_length_per_demo = []
        for start_idx, end_idx in self.ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.max_window_size):
                episode_lookup.append(idx)
            possible_indices = end_idx + 1 - start_idx - self.max_window_size
            max_batched_length_per_demo.append(possible_indices)
        return episode_lookup, max_batched_length_per_demo


class D4RLPlayDataModule(pl.LightningDataModule):
    def __init__(self, goal_type: str, config: DictConfig):
        super().__init__()
        self.env_name = config.env_name
        self.batch_size = config.exp.batch_size
        self.num_workers = config.num_workers
        self.train_size = config.train_size
        self.goal_type = goal_type

        self.min_window_size = config.exp.min_window_size
        self.max_window_size = config.exp.max_window_size
        self.pad = config.exp.pad
        self.include_goal = False if "include_goal" not in config.exp.keys() else config.exp["include_goal"]
        self.goal_sampling_prob = 0 if "goal_sampling_prob" not in config.exp.keys() else config.exp["goal_sampling_prob"]

        env = gym.make(self.env_name)
        self.orig_epi_buffer = env.get_dataset()
        self._obs_dim = self.orig_epi_buffer ['observations'].shape[-1]
        self._action_dim = self.orig_epi_buffer ['actions'].shape[-1]

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        num_epis = self.orig_epi_buffer ['rewards'].shape[0]
        if self.train_size < 1:
            num_train = int(num_epis * self.train_size)
        else:
            num_train = num_epis
        indices = np.arange(num_epis)
        train_indices = indices[:num_train]
        if self.train_size < 1:
            val_indices = indices[num_train: num_epis]
        else:
            # Just use for lightning to call validation_step(), not meaningful to calculate val_loss
            val_indices = indices[int(0.9 * num_train): num_epis]
        train_buffer = {key: value[train_indices] for key, value in self.orig_epi_buffer.items()}
        val_buffer = {key: value[val_indices] for key, value in self.orig_epi_buffer.items()}


        # Instantiate train_dataset
        self.train_dataset = D4RLPlayDataset(
            episode_buffer=train_buffer,
            min_window_size=self.min_window_size,
            max_window_size=self.max_window_size,
            pad=self.pad,
            include_goal=self.include_goal,
            goal_sampling_prob=self.goal_sampling_prob
        )
        self.val_dataset = D4RLPlayDataset(
            episode_buffer=val_buffer,
            min_window_size=self.min_window_size,
            max_window_size=self.max_window_size,
            pad=self.pad,
            include_goal=self.include_goal,
            goal_sampling_prob=self.goal_sampling_prob
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def get_obs_dim(self):
        return self._obs_dim

    def get_action_dim(self):
        return self._action_dim


def main():
    config = OmegaConf.create({
        "env_name": "maze2d-umaze-v1",
        "batch_size": 1024,
        "num_workers": 0,
        "train_size": 0.9,
        "min_window_size": 8,
        "max_window_size": 8,
        "pad": True,
    })
    datamodule = D4RLPlayDataModule(goal_type="state", config=config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        print()


if __name__ == "__main__":
    main()

