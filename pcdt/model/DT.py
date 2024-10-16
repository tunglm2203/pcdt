from typing import Dict, Optional, TypeVar, Tuple
from collections import OrderedDict
from omegaconf import DictConfig

import gym
import d4rl
import wandb
import copy
import random
import math
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdt.model.utils import suppress_output, get_goal, to_tensor, AntMazeVisualizer, record_video, setup_view_for_env, get_1d_sincos_pos_embed_from_grid
from pcdt.model.trajnet_pcdt import SlotMAE_PCDT
from pcdt.model.trajnet import TrajNet


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim, goal_dim, ctx_size, slotmae, model_config):
        super().__init__()
        self.max_ep_len = 1000

        self.slotmae = slotmae
        for param in self.slotmae.parameters():
            param.requires_grad = False
        self.slotmae.eval()


        # MAE specifics
        self.model_config = model_config
        self.embed_dim = model_config.embed_dim
        self.n_head = model_config.n_head
        self.n_enc_layers = model_config.n_enc_layers
        self.pdrop = model_config.pdrop
        self.segment_len = model_config.segment_len
        self.use_pe_as_timestep = model_config.use_pe_as_timestep

        self.GOAL_CONDITION = model_config.goal_condition
        self.ACTION_CONDITION = model_config.action_condition
        if self.ACTION_CONDITION:
            self.n_elements_in_transitions = 3
        else:
            self.n_elements_in_transitions = 2

        blocks = [
            Block(h_dim=self.embed_dim, max_T=3 * ctx_size, n_heads=self.n_head, drop_p=self.pdrop)
            for _ in range(self.n_enc_layers)
        ]
        self.transformer = nn.Sequential(*blocks)

        # Note: You can use Timestep embedding or Positional Encoding
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.embed_dim)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, self.max_ep_len)  # From MaskDP
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0) / 2.0
        self.register_buffer("pos_embed", pe)

        self.embed_state = torch.nn.Linear(obs_dim, self.embed_dim)
        self.embed_action = torch.nn.Linear(act_dim, self.embed_dim)
        self.embed_latent = nn.Linear(self.slotmae.encoder.layers[1].linear2.out_features, self.embed_dim)

        self.embed_ln = nn.LayerNorm(self.embed_dim)

        if self.GOAL_CONDITION:
            self.embed_goal = torch.nn.Linear(goal_dim, self.embed_dim)


        # Note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.embed_dim, obs_dim)
        self.predict_action = nn.Sequential(*([nn.Linear(self.embed_dim, act_dim)] + ([nn.Tanh()])))

    def forward(self, states, actions, goals, traj_mask, timesteps, attention_mask=None):
        
        if not self.GOAL_CONDITION:
            latents, _, _ = self.slotmae.encode(states, goals, traj_mask)
            latents = latents.detach()
        
        batch_size, seq_length = states.shape[0], states.shape[1]
        # if attention_mask is None:
        #     # attention mask for GPT: 1 if can be attended to, 0 if not
        #     attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        if self.ACTION_CONDITION:
            action_embeddings = self.embed_action(actions)
        else:
            action_embeddings = None

        if not self.GOAL_CONDITION:
            # z_emb = self.embed_latent(latents)
            z_emb = latents
        else:
            goals = goals.repeat(1, seq_length, 1)
            z_emb = self.embed_goal(goals)      # NOTE: Goal-condition

        if self.use_pe_as_timestep:
            timesteps = timesteps.to(dtype=torch.long)
            time_embeddings = self.embed_timestep(timesteps)
            state_embeddings = state_embeddings + time_embeddings
            if self.ACTION_CONDITION:
                action_embeddings = action_embeddings + time_embeddings
            z_emb = z_emb + time_embeddings

        if self.ACTION_CONDITION:
            stacked_inputs = torch.stack(
                (z_emb, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, self.n_elements_in_transitions * seq_length, self.embed_dim)
        else:
            stacked_inputs = torch.stack(
                (z_emb, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, self.n_elements_in_transitions * seq_length, self.embed_dim)

        if not self.use_pe_as_timestep:
            stacked_inputs += self.pos_embed[:, :self.n_elements_in_transitions * seq_length, :]

        stacked_inputs = self.embed_ln(stacked_inputs)
        
        x = self.transformer(stacked_inputs)
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, self.n_elements_in_transitions, self.embed_dim).permute(0, 2, 1, 3)

        # get predictions
        if self.ACTION_CONDITION:
            state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
            action_preds = self.predict_action(x[:, 1])  # predict next action given state
        else:
            state_preds = None  # predict next state given state and action
            action_preds = self.predict_action(x[:, 1])  # predict next action given state
        

        return state_preds, action_preds, None
    

    def forward_loss(self, target_s, target_a, target_z, pred_s, pred_a):
        batch_size, T, _ = target_s.shape
        if self.ACTION_CONDITION:
            loss_s = F.mse_loss(pred_s, target_s)
        else:
            loss_s = torch.tensor([0.0])
        loss_a = F.mse_loss(pred_a, target_a)
        return loss_s, loss_a
    


class DTAgent(pl.LightningModule):
    def __init__(
            self,
            seed: int = 1,
            env_name: str = 'antmaze-large-diverse-v2',
            obs_dim: int = 29,
            act_dim: int = 8,
            goal_dim: int = 29,
            goal_type: str = 'state_goal',
            lr: float = 0.0001,
            num_eval_rollouts: int = 50,
            eval_last_k: int = 10,
            base_observation: np.ndarray = None,
            num_epochs: int = 150,
            use_scheduler: bool = True,
            ctx_size: int = None,
            future_horizon: int = 70,
            stage: str = 'pl',
            n_videos_to_render: int = 0,
            model_config: DictConfig = None,
    ):
        super().__init__()

        self.automatic_optimization = False  # Perform manual optimization

        ckpt = torch.load(model_config.tjn_ckpt_path, map_location=self.device)
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            key = k.replace('model.', '')
            state_dict[key] = v
        assert ckpt['hyper_parameters']['env_name'] == env_name
        assert model_config['pretrain_use_new_stack_so'] == ckpt['hyper_parameters']['model_config']['use_new_stack_so']
        assert model_config['pretrain_padding_zeros'] == ckpt['hyper_parameters']['model_config']['padding_zeros']

        slotmae = SlotMAE_PCDT(obs_dim=obs_dim, action_dim=act_dim, goal_dim=goal_dim, ctx_size=ctx_size, config=ckpt['hyper_parameters']['model_config'])
        slotmae.load_state_dict(state_dict)

        # trajnet = TrajNet.load_from_checkpoint(checkpoint_path=model_config.tjn_ckpt_path, map_location=self.device, num_epochs=None)
        # slotmae = trajnet.model

        self.seed = seed
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.ctx_size = ctx_size
        self.future_horizon = future_horizon
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.num_epochs = num_epochs
        self.eval_last_k = eval_last_k
        self.goal_type = goal_type
        self.num_eval_rollouts = num_eval_rollouts
        self.stage = stage
        self.base_observation = base_observation

        self.mask_ratios = model_config.mask_ratios
        self.mask_ratio_weights = model_config.mask_ratio_weights
        self.model_config = model_config

        self.render_size = 300
        self.n_videos_to_render = n_videos_to_render
        self.eval_score = []

        with suppress_output():
            _env = gym.make(self.env_name)
            if 'antmaze' in self.env_name:
                self.eval_env = AntMazeVisualizer(_env, env_name)
            else:
                self.eval_env = _env
            self.eval_env.seed(seed=self.seed)
            self.eval_env.action_space.seed(seed=self.seed)
            self.eval_env.observation_space.seed(seed=self.seed)
            if 'antmaze' in self.env_name:
                setup_view_for_env(self.eval_env, self.env_name, self.render_size)

        self.model = DecisionTransformer(self.obs_dim, self.act_dim, self.goal_dim, self.ctx_size, slotmae, model_config)
        self.save_hyperparameters()

    def ar_mask(self, batch_size: int, length: int, keep_len: float, device):
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        return mask

    def training_step(self, batch, batch_idx, log_prefix='train'):
        opt = self.optimizers() if log_prefix == 'train' else None
        observations, actions, goals, _, _, timesteps = batch
        batch_size, length, _ = observations.shape

        # traj_mask_ratio = np.random.choice(self.mask_ratios, 1, p=self.mask_ratio_weights)[0]   # 0 - keep, 1 - remove
        # keep_len = max(1, int(np.ceil(self.ctx_size * (1 - traj_mask_ratio))))
        keep_len = self.ctx_size
        traj_obs_mask = self.ar_mask(batch_size, length, keep_len, observations.device)

        # attention_mask = torch.cat([torch.zeros(self.ctx_size - observations.shape[1]), torch.ones(observations.shape[1])])
        # attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        # attention_mask =  attention_mask.expand(batch_size, -1)
        
        pred_s, pred_a, gt_z = self.model.forward(observations, actions, goals, traj_obs_mask, timesteps) # DT forward
        state_loss, action_loss = self.model.forward_loss(observations, actions, gt_z, pred_s, pred_a)
        if self.model_config.action_condition:
            loss = state_loss + action_loss
        else:
            loss = action_loss

        # Perform optimization step
        if log_prefix == 'train':
            opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            opt.step()

            if self.use_scheduler and 'kitchen' not in self.env_name:
                lr_sch = self.lr_schedulers()
                lr_sch.step()

        log_metrics = {
            f"{log_prefix}/action_loss": action_loss.item(),
        }
        if self.model_config.action_condition:
            log_metrics.update({f"{log_prefix}/state_loss": state_loss.item()})

        self.log_dict(log_metrics)

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, log_prefix='val')

    def do_env_render(self, with_goal=True, is_render=True, info=None):
        sub_goal = None
        if info is not None:
            if 'sub_goal' in info:
                sub_goal = info['sub_goal']
        if sub_goal is not None:
            assert isinstance(sub_goal, torch.Tensor), f"type sub_goal={type(sub_goal)}"
            if len(sub_goal.shape) == 2:
                sub_goal_viz = sub_goal[0][:2].tolist()
            elif len(sub_goal.shape) == 1:
                sub_goal_viz = sub_goal[:2].tolist()
            else:
                raise NotImplementedError
        else:
            sub_goal_viz = None
        return self.eval_env.render(mode='rgb_array', width=self.render_size, height=self.render_size, with_goal=with_goal, sub_goal=sub_goal_viz) if is_render else None


    # eval funcs
    def init_eval(self, ini_obs, goal, obs_dim: int, action_dim: int, goal_dim: int):
        # ini_obs: (obs_dim, )
        observations = torch.zeros(size=(1, self.ctx_size, obs_dim), device=self.device)  # 1 x ctx_size x obs_dim
        observations[0, 0] = ini_obs
        # goal: (goal_dim, )
        goal = goal.view(1, -1, goal_dim)
        actions = torch.zeros(size=(1, self.ctx_size, action_dim), device=self.device)  # 1 x ctx_size x action_dim
        return observations, actions, goal

    def make_ar_mask(self, timestep: int):
        '''
        make observation mask for first autoregressive ctx_size steps
        '''
        obs_mask = torch.zeros(size=(1, self.ctx_size), device=self.device)
        action_mask = torch.zeros(size=(1, self.ctx_size), device=self.device)
        action_mask[:, -1] = 1  # the last action is always masked
        if timestep < self.ctx_size - 1:
            obs_mask[0, 1 + timestep:] = 1  # at first ctx_size steps, replace idle observations with obs masked token
            action_mask[0, timestep:] = 1
        return obs_mask, action_mask

    def ar_step(self, timestep: int, observations, actions, goal):
        obs_mask, _ = self.make_ar_mask(timestep)
        timesteps = torch.arange(timestep + 1, dtype=torch.long).view(1, timestep + 1)[:, -self.ctx_size:]
        timesteps = torch.cat([timesteps, torch.zeros((1, self.ctx_size - timesteps.shape[1]))], dim=1).to(self.device)
        _, pred_a, _ = self.model.forward(observations, actions, goal, obs_mask, timesteps)

        if timestep < self.ctx_size - 1:
            action = pred_a[0, timestep]
        else:
            action = pred_a[0, -1]
        return action

    def ar_step_end(self, timestep: int, next_obs, action, obs_seq, action_seq):
        if timestep < self.ctx_size - 1:
            obs_seq[0, 1 + timestep] = next_obs
            new_obs_seq = obs_seq
            action_seq[0, timestep] = action
            new_action_seq = action_seq
        else:
            next_obs = next_obs.view(1, 1, obs_seq.shape[-1])
            new_obs_seq = torch.cat([obs_seq, next_obs], dim=1)
            action_seq[0, -1] = action
            new_action_seq = torch.cat([action_seq, torch.zeros(size=(1, 1, action_seq.shape[-1]), device=action_seq.device)], dim=1)

        new_obs_seq = new_obs_seq[:, -self.ctx_size:]   # moving ctx
        new_action_seq = new_action_seq[:, -self.ctx_size:]
        return new_obs_seq, new_action_seq

    def on_validation_epoch_end(self):
        if self.current_epoch > 0 and self.use_scheduler and 'kitchen' in self.env_name:
            lr_sch = self.lr_schedulers()
            lr_sch.step()

        if self.current_epoch + 1 > self.num_epochs - self.eval_last_k:  # start evaluations for the last k checkpoints
            eval_horizon = self.eval_env._max_episode_steps

            renders, ep_lengths = [], []
            rollout_scores = torch.zeros(size=(self.num_eval_rollouts,), device=self.device)
            for i in tqdm(range(self.num_eval_rollouts), leave=False):
                is_render = i < self.n_videos_to_render  # Flag to check if render

                render_per_episode, done, ep_length = [], False, 0
                with suppress_output():
                    ini_obs = self.eval_env.reset()
                    goal = get_goal(env=self.eval_env, goal_type=self.goal_type, obs=ini_obs)

                # Override goal from 1st obs in dataset
                if self.goal_type == 'state_goal' and 'antmaze' in self.env_name:
                    assert hasattr(self, 'base_observation')
                    obs_goal = self.base_observation.copy()
                    obs_goal[:2] = goal
                    goal = obs_goal
                elif self.goal_type == 'state_goal' and 'kitchen' in self.env_name:
                    goal = copy.deepcopy(goal)  # Use goal style from pcdt

                render_per_episode.append(self.do_env_render(is_render=is_render))

                ini_obs = to_tensor(ini_obs, self.device)[:self.obs_dim]
                goal = to_tensor(goal, self.device)
                obs, actions, goal = self.init_eval(ini_obs, goal, self.obs_dim, self.act_dim, self.goal_dim)
                for t in range(eval_horizon):
                    a = self.ar_step(t, obs, actions, goal)
                    next_obs, reward, done, _ = self.eval_env.step(a.cpu().numpy())
                    rollout_scores[i] += reward
                    render_per_episode.append(self.do_env_render(is_render=is_render))
                    ep_length += 1

                    if done:
                        break

                    next_obs = to_tensor(next_obs, self.device)[:self.obs_dim]
                    obs, actions = self.ar_step_end(t, next_obs, a, obs, actions)

                if is_render:
                    renders.append(np.array(render_per_episode))
                rollout_scores[i] = self.eval_env.get_normalized_score(rollout_scores[i])
                ep_lengths.append(ep_length)

            self.eval_score.append(rollout_scores.mean().item())

            self.log_dict({
                'eval/norm_score': self.eval_score[-1]
            })
            if self.n_videos_to_render > 0:
                video = record_video(renders=renders, n_cols=self.n_videos_to_render)
                wandb.log({'video': video}, step=self.global_step)

        if self.current_epoch + 1 == self.num_epochs:  # get the best evaluation result among the last k checkpoints
            self.log_dict({
                'result/norm_score': np.max(self.eval_score[-self.eval_last_k:]),
            })

    def configure_optimizers(self):

        # optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer_dict = {
            'optimizer': optimizer,
        }
        if 'kitchen' in self.env_name:
            if self.use_scheduler:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5, step_size=self.num_epochs // 2)
                optimizer_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}})
        else:
            warmup_steps = 10000
            if self.use_scheduler:
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda steps: min((steps + 1) / warmup_steps, 1)
                )
                optimizer_dict.update({'lr_scheduler': scheduler})

        return optimizer_dict