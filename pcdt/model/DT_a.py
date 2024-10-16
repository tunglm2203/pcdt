from collections import OrderedDict

from typing import Dict, Optional, TypeVar, Tuple
from omegaconf import DictConfig

import gym
import d4rl
import wandb
import copy
import random
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from pcdt.model.utils import suppress_output, get_goal, to_tensor, AntMazeVisualizer, record_video
from pcdt.data.sequence import D4RLDataModule_StateGoal
#from pcdt.model.trajnet import TrajNet
from pcdt.model.trajnet_a import TrajNetWithAction
### For Decision Transformer ###
#home/ldh/anaconda3/envs/pcdt/lib/python3.7/site-packages/transformers/__init__.py
import transformers
from pcdt.model.trajectory_gpt2 import GPT2Model

from pcdt.model.utils import get_1d_sincos_pos_embed_from_grid
### For Decision Transformer ###
# time embeddings are treated similar to positional embeddings
import math
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
        #breakpoint()
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
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
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

class DecisionTransformer(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    ### Same structure of MTM's Init function 
    
    # DT origin :  
    # max_ep_len= 1000 
    # embed_dim =128 
    def __init__(self, obs_dim, act_dim, slotmae, model_config):
        super().__init__()
        self.max_ep_len=1000
        #model_config.segment_len * 2  # x2 since (s, a) is tuple of 2 items
        

        self.slotmae = slotmae
        for param in self.slotmae.parameters():
            param.requires_grad = False
        self.slotmae.eval()

        # MAE specifics
        self.model_config = model_config
        self.embed_dim = model_config.embed_dim
        self.n_head = model_config.n_head
        self.n_enc_layers = model_config.n_enc_layers
        self.n_dec_layers = model_config.n_dec_layers
        self.pdrop = model_config.pdrop
        self.segment_len = model_config.segment_len
        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=self.embed_dim,
            n_head=4,
            n_layer=3
        )
        '''
        {
        "activation_function": "gelu_new",
        "attn_pdrop": 0.1,
        "bos_token_id": 50256,
        "embd_pdrop": 0.1,
        "eos_token_id": 50256,
        "gradient_checkpointing": false,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "n_embd": 256,
        "n_head": 12,
        "n_inner": null,
        "n_layer": 12,
        "n_positions": 1024,
        "resid_pdrop": 0.1,
        "summary_activation": null,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": true,
        "summary_type": "cls_index",
        "summary_use_proj": true,
        "transformers_version": "4.5.1",
        "use_cache": true,
        "vocab_size": 1
        }

        '''

        
        #self.transformer = GPT2Model(config)
        blocks = [Block(h_dim=self.embed_dim, max_T=30, n_heads=1, drop_p=0.1) for _ in range(3)]
        self.transformer = nn.Sequential(*blocks)
        # self.embed_dim : 256. (same as MTM)
        #breakpoint()
        # note: You can use Timestep embedding or Positional Encoding
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.embed_dim)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, self.max_ep_len)  # From MaskDP
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0) / 2.0
        self.register_buffer("pos_embed", pe)


        #self.embed_return = torch.nn.Linear(1, self.embed_dim)
        self.embed_state = torch.nn.Linear(obs_dim, self.embed_dim)
        self.embed_action = torch.nn.Linear(act_dim, self.embed_dim)
        ######################    For BottleNeck   ###################### 
        self.reconstruct_latent = False
        self.latent_embed = nn.Linear(self.embed_dim , self.embed_dim)
        ######################    For BottleNeck   ######################   
        self.embed_ln = nn.LayerNorm(self.embed_dim)
        
        assert self.slotmae.embed_dim == self.embed_dim
        
        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.embed_dim, obs_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.embed_dim, act_dim)] + ([nn.Tanh()]))
        )
        #self.predict_return = torch.nn.Linear(self.embed_dim, 1)
        self.predict_latent = torch.nn.Linear(self.embed_dim, self.slotmae.n_slots)  # n_slots dimension : 4

        self.use_mode_encoding = False
        if self.use_mode_encoding:
            self.mode_encoding_state = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.mode_encoding_action = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.mode_decoding_state = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.mode_decoding_action = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

            self.mode_encoding_latent = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.mode_decoding_latent = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            # self.mode_encoding_goal = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            # self.mode_decoding_goal = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        


        
    # def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
    def forward(self, states, actions, goals, traj_mask,action_mask , timesteps, attention_mask=None):
        
        latents, _, _ ,_= self.slotmae.encode(states, actions,goals,traj_mask,action_mask)
        
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = F.relu(self.embed_state(states))
        action_embeddings = F.relu(self.embed_action(actions))
        #returns_embeddings = self.embed_return(returns_to_go)

        
        z_emb = latents
        #z_emb = latents

        if self.use_mode_encoding:
            state_embeddings = state_embeddings + self.mode_encoding_state
            action_embeddings = action_embeddings + self.mode_encoding_action
            z_emb = z_emb + self.mode_encoding_latent
        
        # time embeddings are treated similar to positional embeddings
        #timesteps = timesteps.to(dtype=torch.long)
        
        #time_embeddings = self.embed_timestep(timesteps)
        
        # time embeddings are treated similar to positional embeddings
        #state_embeddings = state_embeddings + time_embeddings
        #action_embeddings = action_embeddings + time_embeddings
        #z_emb=  z_emb +   time_embeddings
        #returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        
        #stacked_inputs = torch.stack(
        #    (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        

        ########################### 10 token stack setting #########################
        
        stacked_inputs = torch.stack(
            (z_emb, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.embed_dim)
        stacked_inputs += self.pos_embed[:, :3*seq_length, :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        '''
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        stacked_attention_mask=stacked_attention_mask.to(dtype=torch.long, device=states.device)        
        
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        '''
        x=self.transformer(stacked_inputs)
        x = x.reshape(batch_size, seq_length, 3, self.embed_dim).permute(0, 2, 1, 3)
        # get predictions
        
        #return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        
        ########################### 10 token stack setting #########################



        ########################### 4 token stack setting #########################    
        '''
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.embed_dim)  
        stacked_inputs = torch.cat([z_emb, stacked_inputs], dim=1)  # First 4 tokens are bottleneck

        seq_len = self.segment_len * 2 + self.slotmae.n_slots
        stacked_inputs += self.pos_embed[:, :seq_len, :]
        stacked_inputs = self.embed_ln(stacked_inputs)        
        
        # add pos embed

        # to make the attention mask fit the stacked inputs, have to stack it as well        
        slotmask=torch.ones(batch_size,self.slotmae.n_slots)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1)
        stacked_attention_mask=stacked_attention_mask.permute(0, 2, 1).reshape(batch_size, self.segment_len * 2)
        stacked_attention_mask=torch.cat([slotmask, stacked_attention_mask], dim=1)
        stacked_attention_mask=stacked_attention_mask.to(dtype=torch.long, device=states.device)

        ''
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        stacked_attention_mask=stacked_attention_mask.to(dtype=torch.long, device=states.device)
        ''
        #breakpoint()  
        # we feed in the input embeddings (not word indices as in NLP) to the model
        '
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        '
        x=self.transformer(stacked_inputs)
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        
        
        #x=x.reshape(batch_size,24,self.embed_dim)
        #latent_parts=x[:, :4, :]
        #s_a_parts= x[:, 4:24, :]
        #latent_parts=latent_parts.unsqueeze(2).permute(0, 2, 1, 3) # 1024,1,4,256
        #s_a_parts=s_a_parts.reshape(batch_size, 10, 2, self.embed_dim).permute(0, 2, 1, 3) # 1024,2,10,256
        #s_a_parts=torch.cat([latent_parts,s_a_parts],dim=1)
        # get predictions
        #return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        #  states (0), or actions (1); i.e. x[:,1,t] is the token for s_t
        #latent_preds=self.predict_latent(latent_parts[:,2])
        
        # 20240208
        s_a_parts= x[:, 4:24, :]
        s_a_parts=s_a_parts.reshape(batch_size, seq_length, 2, self.embed_dim).permute(0, 2, 1, 3) # 1024,2,10,256
        #s_a_parts=x.reshape(batch_size, 10, 2, self.embed_dim).permute(0, 2, 1, 3)
        state_preds = self.predict_state(s_a_parts[:,1])    # predict next state given state and action
        action_preds = self.predict_action(s_a_parts[:,0])  # predict next action given state
        '''
        return state_preds, action_preds, latents
    

    def forward_loss(self, target_s, target_a, target_z, pred_s, pred_a):
        batch_size, T, _ = target_s.shape
        loss_s = F.mse_loss(pred_s, target_s)
        loss_a = F.mse_loss(pred_a, target_a)
        #if self.reconstruct_latent:
        #    loss_z = F.mse_loss(pred_z, target_z)
        #else:
        #    loss_z = torch.tensor([0.0])
        return loss_s, loss_a
    


class DTAgent(pl.LightningModule):
    def __init__(
            self,
            seed: int = 1,
            env_name: str = 'antmaze-medium-diverse-v2',
            obs_dim: int = 29,
            act_dim: int = 8,
            goal_dim: int = 29,
            goal_type: str = 'state_goal',
            lr: float = 0.0001,
            num_eval_rollouts: int = 50,
            eval_last_k: int = 10,
            base_observation: np.ndarray = None,
            num_epochs: int = 150,
            use_scheduler: bool = False,
            ctx_size: int = None,
            future_horizon: int = 70,
            stage: str = 'pl',
            model_config: DictConfig = None,
    ):
        super().__init__()

        self.automatic_optimization = False  # Perform manual optimization

        trajnet = TrajNetWithAction.load_from_checkpoint(checkpoint_path=model_config.tjn_ckpt_path, map_location=self.device, num_epochs=None)
        slotmae = trajnet.model

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
        #self.policy_mask_ratios = model_config.policy_mask_ratios

        self.data_shapes = OrderedDict()
        self.data_shapes['states'] = np.array((1, obs_dim))
        self.data_shapes['actions'] = np.array((1, act_dim))

        self.render_size = 300
        self.n_videos_to_render = 2
        self.eval_score = []

        with suppress_output():
            _env = gym.make(self.env_name)
            self.eval_env = AntMazeVisualizer(_env, env_name)
            self.eval_env.seed(seed=self.seed)
            self.eval_env.action_space.seed(seed=self.seed)
            self.eval_env.observation_space.seed(seed=self.seed)
            setup_view_for_env(self.eval_env, self.env_name, self.render_size)

        self.model = DecisionTransformer(self.obs_dim, self.act_dim, slotmae, model_config)

    def ar_mask(self, batch_size: int, length: int, keep_len: float, device):
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        return mask

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        observations, actions, goals, _, _,timesteps = batch
        batch_size, length, _ = observations.shape

        traj_mask_ratio = np.random.choice(self.mask_ratios, 1, p=self.mask_ratio_weights)[0]   # 0 - keep, 1 - remove
        # keep_len = max(1, int(self.ctx_size * np.ceil(1 - traj_mask_ratio)))
        keep_len = self.ctx_size
        traj_obs_mask = self.ar_mask(batch_size, length, keep_len, observations.device)
        action_mask = self.ar_mask(batch_size, length, keep_len - 1, actions.device)
        #policy_mask_ratio = np.random.choice(self.policy_mask_ratios)
        #policy_mask = create_random_masks(self.data_shapes, policy_mask_ratio, self.ctx_size, self.device)     # 1 - keep, 0 - remove
        #policy_mask = convert_mask_to_pcdt_mask(policy_mask)  # 0 - keep, 1 - remove
        attention_mask = torch.cat([torch.zeros(self.ctx_size-observations.shape[1]), torch.ones(observations.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        attention_mask=  attention_mask.expand(batch_size, -1)
        
        #def forward(self, states, actions, goals, traj_mask, timesteps, attention_mask=None):
        pred_s, pred_a, gt_z = self.model.forward(observations, actions, goals, traj_obs_mask,action_mask ,timesteps,attention_mask=attention_mask) # DT forward
        state_loss, action_loss = self.model.forward_loss(observations, actions, gt_z, pred_s, pred_a)


        loss = state_loss + action_loss

        # Perform optimization step
        opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        opt.step()

        if self.use_scheduler:
            lr_sch = self.lr_schedulers()
            lr_sch.step()

        log_metrics = {
            "train/state_loss": state_loss.item(),
            "train/action_loss": action_loss.item(),
        }


        self.log_dict(log_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        observations, actions, goals, _, _,timesteps = batch
        batch_size, length, _ = observations.shape

        traj_mask_ratio = np.random.choice(self.mask_ratios, 1, p=self.mask_ratio_weights)[0]  # 0 - keep, 1 - remove
        keep_len = max(1, int(self.ctx_size * (1 - traj_mask_ratio)))
        traj_obs_mask = self.ar_mask(batch_size, length, keep_len, observations.device)
        action_mask = self.ar_mask(batch_size, length, keep_len - 1, actions.device)

        #policy_mask_ratio = np.random.choice(self.policy_mask_ratios)
        #policy_mask = create_random_masks(self.data_shapes, policy_mask_ratio, self.ctx_size, self.device)  # 1 - keep, 0 - remove
        #policy_mask = convert_mask_to_pcdt_mask(policy_mask)  # 0 - keep, 1 - remove
        attention_mask = torch.cat([torch.zeros(self.ctx_size-observations.shape[1]), torch.ones(observations.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        attention_mask=  attention_mask.expand(batch_size, -1)

        #print("!!!")
        #breakpoint()
        pred_s, pred_a, gt_z = self.model.forward(observations, actions, goals, traj_obs_mask,action_mask,timesteps,attention_mask=attention_mask)  # Encode/Decode
        state_loss, action_loss= self.model.forward_loss(observations, actions, gt_z, pred_s, pred_a)

        log_metrics = {
            "val/state_loss": state_loss.item(),
            "val/action_loss": action_loss.item(),
        }


        self.log_dict(log_metrics)

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
        traj_obs_mask, action_mask = self.make_ar_mask(timestep)
        batch_size, length, _ = observations.shape
        #policy_mask = {}
        #policy_mask['states'] = traj_obs_mask
        #policy_mask['actions'] = action_mask
        #policy_mask = reshape_mask(policy_mask)
        attention_mask = torch.cat([torch.zeros(self.ctx_size-observations.shape[1]), torch.ones(observations.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        attention_mask=  attention_mask.expand(batch_size, -1)
        #print(attention_mask)
        # Note: just for learning (we use positional embeding now.)
        timesteps=1
        _, pred_a, _ = self.model.forward(observations, actions, goal, traj_obs_mask, action_mask,timesteps, attention_mask = attention_mask)
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
        if self.current_epoch + 1 > self.num_epochs - self.eval_last_k:  # start evaluations for the last k checkpoints
            eval_horizon = self.eval_env.env._max_episode_steps

            renders, ep_lengths = [], []
            rollout_scores = torch.zeros(size=(self.num_eval_rollouts,), device=self.device)
            for i in tqdm(range(self.num_eval_rollouts), leave=False):
                is_render = i < self.n_videos_to_render  # Flag to check if render

                render_per_episode, done, ep_length = [], False, 0
                with suppress_output():
                    ini_obs = self.eval_env.reset()
                    goal = get_goal(env=self.eval_env, goal_type=self.goal_type, obs=ini_obs)

                if self.goal_type == 'state_goal' and 'antmaze' in self.env_name:
                    assert hasattr(self, 'base_observation')
                    obs_goal = self.base_observation.copy()
                    obs_goal[:2] = goal
                    goal = obs_goal
                else:
                    raise NotImplementedError
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
            video = record_video(renders=renders)
            wandb.log({'video': video}, step=self.global_step)

        if self.current_epoch + 1 == self.num_epochs:  # get the best evaluation result among the last k checkpoints
            self.log_dict({
                'result/norm_score': np.max(self.eval_score[-self.eval_last_k:]),
            })

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        optimizer_dict = {
            'optimizer': optimizer,
        }
        if self.use_scheduler:
            raise NotImplementedError

        return optimizer_dict