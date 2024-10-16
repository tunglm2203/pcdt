import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor, device
from typing import Union
import numpy as np
import math
from omegaconf import DictConfig
from .policylm import PolicyLM

Device = Union[device, str, int, None]


class SlotMAEPE(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, slots: Tensor, observations: Tensor, actions: Tensor, goal: Tensor = None) -> Tensor:
        """
        Args:
            [batch_size, seq_len, embedding_dim]
        """
        assert observations.shape[1] == actions.shape[1]
        if goal is not None:  # for encoder
            slots += self.pe[:, :slots.shape[1]]
            goal += self.pe[:, slots.shape[1]:slots.shape[1] + goal.shape[1]]
            observations += self.pe[:, slots.shape[1] + goal.shape[1]: slots.shape[1] + goal.shape[1] + observations.shape[1]]
            actions += self.pe[:, slots.shape[1] + goal.shape[1]: slots.shape[1] + goal.shape[1] + actions.shape[1]]
            return slots, goal, observations, actions
        else:  # for decoder
            slots += self.pe[:, :slots.shape[1]]
            observations += self.pe[:, slots.shape[1]: slots.shape[1] + observations.shape[1]]
            actions += self.pe[:, slots.shape[1]: slots.shape[1] + actions.shape[1]]
            return slots, observations, actions



class SlotMAEWithAction(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, goal_dim: int, config: DictConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.embed_dim = config.embed_dim
        self.n_slots = config.n_slots

        self.positional_encoding = SlotMAEPE(d_model=self.embed_dim)
        self.slots = nn.Embedding(self.n_slots, self.embed_dim)
        # encoder
        self.obs_embed = nn.Linear(self.obs_dim, self.embed_dim)
        self.goal_embed = nn.Linear(self.goal_dim, self.embed_dim)
        self.action_embed = nn.Linear(self.action_dim, self.embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            dim_feedforward=self.embed_dim * 4,
            nhead=config.n_head,
            dropout=config.pdrop,
            activation=F.gelu,
            norm_first=True,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, config.n_enc_layers)
        self.encoder_norm = nn.LayerNorm(self.embed_dim)

        # decoder

        self.obs_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.action_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.obs_decode_emb = nn.Linear(self.embed_dim, self.embed_dim)
        self.action_decode_emb = nn.Linear(self.embed_dim, self.embed_dim)

        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            dim_feedforward=self.embed_dim * 4,
            nhead=config.n_head,
            dropout=config.pdrop,
            activation=F.gelu,
            norm_first=True,
            batch_first=True
        )

        self.decoder = nn.TransformerEncoder(self.decoder_layer, config.n_dec_layers)

        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        self.obs_decoder = nn.Linear(self.embed_dim, self.obs_dim)
        self.action_decoder = nn.Linear(self.embed_dim, self.action_dim)
    
    def encode(self, observations: Tensor, actions: Tensor, goal: Tensor, obs_mask: Tensor, action_mask: Tensor):
        batch_size, length, _ = observations.shape
        slots = self.slots(torch.arange(self.n_slots, device=observations.device)).repeat(batch_size, 1, 1)

        # goal: B X 1 x goal_dim
        goal_embeddings = self.goal_embed(goal)
        # obs: B X L x obs_dim
        obs_embeddings = self.obs_embed(observations)
        # actions: B X L x action_dim
        action_embeddings = self.action_embed(actions)

        s, g, o, a = self.positional_encoding(slots, obs_embeddings, action_embeddings, goal_embeddings)
        o_keep = o[obs_mask == 0].view(batch_size, -1, self.embed_dim)   # 1024 10 256
        a_keep = a[action_mask == 0].view(batch_size, -1, self.embed_dim)# 1024 9 256
        
        s_o_a_keep=torch.stack((s[:,:-1,:],o_keep[:,:-1,:],a_keep),dim=1).permute(0, 2, 1, 3).reshape(batch_size,-1, self.embed_dim)
        s_o_a_keep=torch.cat([s_o_a_keep, s[:,-1,:].unsqueeze(1), o_keep[:,-1,:].unsqueeze(1)],dim=1)
    
        
        '''
        s=s.repeat(1,o.shape[1]/s.shape[1],1)
        slot_mask=obs_mask
        stacked_s_o_a=torch.stack((s,o,a), dim=1).permute(0, 2, 1, 3).reshape(batch_size,-1, self.embed_dim)#
        total_mask = torch.stack([slot_mask,obs_mask,action_mask]).permute(0,2,1).reshape(batch_size,-1)
        stacked_s_o_a = stacked_s_o_a[total_mask == 0].view(batch_size, -1, self.embed_dim)
        '''
        enc_inputs = torch.cat([g, s_o_a_keep], dim=1)
        encoded_keep = self.encoder(enc_inputs)
        encoded_keep = self.encoder_norm(encoded_keep)
        encoded_goal = encoded_keep[:, :g.shape[1],:]
        bottleneck = encoded_keep[:,g.shape[1] ::3,:]
        
        encoded_obs = encoded_keep[:,g.shape[1]+1 ::3,:]
        encoded_action =  encoded_keep[:,g.shape[1]+2 ::3,:]
        return bottleneck, encoded_goal, encoded_obs, encoded_action
    
    def decode(self, bottleneck: Tensor, obs_mask: Tensor, action_mask: Tensor):
        batch_size = bottleneck.shape[0]
        mask_obs = self.obs_mask_token.repeat(obs_mask.shape[0], obs_mask.shape[1], 1)
        mask_actions = self.action_mask_token.repeat(action_mask.shape[0], action_mask.shape[1], 1)

        b, o, a = self.positional_encoding(bottleneck, mask_obs, mask_actions)
        dec_inputs = torch.cat([b, o, a], dim=1)
        decode_out = self.decoder(dec_inputs)
        decode_out = self.decoder_norm(decode_out)

        obs_out = decode_out[:, b.shape[1]::2, :]
        action_out = decode_out[:, b.shape[1]+1::2, :]

        pred_o = self.obs_decoder(obs_out)
        pred_a = self.action_decoder(action_out)

        return pred_o, pred_a

    def forward(self, observations: Tensor, actions: Tensor, goal: Tensor, obs_mask: Tensor, action_mask: Tensor):
        '''
        obs_mask: boolean tensor, True means masked
        '''
        bottleneck, _, _, _ = self.encode(observations, actions, goal, obs_mask, action_mask)
        pred_o, pred_a = self.decode(bottleneck, obs_mask, action_mask)

        return pred_o, pred_a
    

#class TrajNetWithAction(PolicyLM):
class TrajNetWithAction(pl.LightningModule):
    def __init__(self, seed: int, env_name: str, obs_dim: int, action_dim: int, goal_dim: int, lr: float, use_scheduler: bool, num_epochs: int, ctx_size: int, future_horizon: int, stage: str, num_eval_rollouts: int, eval_last_k: int, goal_type: str, model_config: DictConfig, **kwargs):
        super().__init__()
        self.env_name = env_name
        self.ctx_size = ctx_size
        self.future_horizon = future_horizon
        self.stage = stage
        self.mask_type = model_config.mask_type
        self.ar_mask_ratios = model_config.ar_mask_ratios
        self.rnd_mask_ratios = model_config.rnd_mask_ratios
        self.ar_mask_ratio_weights = model_config.ar_mask_ratio_weights
        self.traj_len = self.ctx_size + self.future_horizon
        self.lr = lr
        self.num_epochs = num_epochs
        self.goal_type = goal_type
        self.model = SlotMAEWithAction(obs_dim, action_dim, goal_dim, model_config)

        self.save_hyperparameters()
    
    def forward(self, observations: Tensor, actions: Tensor, goal: Tensor, obs_mask: Tensor, action_mask: Tensor):
        return self.model.forward(observations, actions, goal, obs_mask, action_mask)

    def loss(self, target_o: Tensor, target_a: Tensor, pred_o: Tensor, pred_a: Tensor, padding_mask: Tensor):
        B, T = padding_mask.size()
        padding_mask = padding_mask.float()

        # need to check 
        if self.goal_type == 'state' or self.goal_type == 'state_goal':
            pred_o = pred_o[:, :, :self.model.goal_dim]
            target_o = target_o[:, :, :self.model.goal_dim]
        
        loss_o = F.mse_loss(pred_o, target_o, reduction='none') # B x L x D
        loss_o *= 1 - padding_mask.view(B, T, -1)  # padding mask: 0 keep, 1 pad, so 1 - padding_mask needed
        loss_o = loss_o.mean() * (B * T) / (1 - padding_mask).sum()

        loss_a = F.mse_loss(pred_a, target_a, reduction='none') # B x L x D
        loss_a *= 1 - padding_mask.view(B, T, -1)  # padding mask: 0 keep, 1 pad, so 1 - padding_mask needed
        loss_a = loss_a.mean() * (B * T) / (1 - padding_mask).sum()

        return loss_o, loss_a
    
    def ar_mask(self, batch_size: int, length: int, keep_len: float, device: Device):
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        return mask

    def rnd_mask(self, batch_size: int, length: int, mask_ratio: float, device: Device):
        if length > 0:
            keep_len = max(1, int(length * (1 - mask_ratio)))  # at least keep the first obs
        else:
            keep_len = 0

        noise = torch.rand(size=(batch_size, length), device=device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is mask
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def training_step(self, batch, batch_idx):
        observations, actions, goal, valid_length, padding_mask,_ = batch
        batch_size, _, _ = observations.shape
        ar_mask_ratio = np.random.choice(self.ar_mask_ratios, 1, p=self.ar_mask_ratio_weights)[0]
        rnd_mask_ratio = np.random.choice(self.rnd_mask_ratios, 1)[0]
        # For convenience, make sure the number of unmasked (obs/action) is the same across examples when masking
        if self.mask_type == 'mae_all':
            observations = observations[:, :self.traj_len]
            actions = actions[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            obs_mask = self.rnd_mask(batch_size, obs_length, rnd_mask_ratio, observations.device)
            action_mask = self.rnd_mask(batch_size, obs_length, rnd_mask_ratio, actions.device)
        elif self.mask_type == 'ae_h':
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            # padding history actions as the same length of history obs
            actions = torch.cat([actions[:, :keep_len - 1], torch.zeros(batch_size, 1, actions.shape[-1], device=actions.device)], dim=1)
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.ar_mask(batch_size, keep_len, keep_len, observations.device)
            action_mask = self.ar_mask(batch_size, keep_len, keep_len, actions.device)
        elif self.mask_type == 'mae_h':
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            # padding history actions as the same length of history obs
            actions = torch.cat([actions[:, :keep_len - 1], torch.zeros(batch_size, 1, actions.shape[-1], device=actions.device)], dim=1)
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
            action_mask = self.rnd_mask(batch_size, keep_len - 1, rnd_mask_ratio, actions.device)
            action_mask = torch.cat([action_mask, torch.zeros(batch_size, 1, device=action_mask.device)], dim=1)  # doesn't hope to infer the padding token
        elif self.mask_type == 'mae_f':
            observations = observations[:, :self.traj_len]
            actions = actions[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = self.ctx_size  # match the beginning of eval (obs length is less than ctx size)
            obs_mask = self.ar_mask(batch_size, obs_length, keep_len, observations.device)
            action_mask = self.ar_mask(batch_size, obs_length, keep_len - 1, observations.device)
        elif self.mask_type == 'mae_rc':
            observations = observations[:, :self.traj_len]
            actions = actions[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))
            history_obs_rnd_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
            history_action_rnd_mask = self.rnd_mask(batch_size, keep_len - 1, rnd_mask_ratio, actions.device)
            history_action_rnd_mask = torch.cat([history_action_rnd_mask, torch.ones(batch_size, 1, device=history_action_rnd_mask.device)], dim=1)
            future_causal_mask = self.ar_mask(batch_size, obs_length - keep_len, 0, observations.device)
            obs_mask = torch.cat([history_obs_rnd_mask, future_causal_mask], dim=1)
            action_mask = torch.cat([history_action_rnd_mask, future_causal_mask], dim=1)
        else:
            raise NotImplementedError

        pred_o, pred_a = self(observations, actions, goal, obs_mask, action_mask)
        loss_o, loss_a = self.loss(observations, actions, pred_o, pred_a, padding_mask)

        loss = loss_o + loss_a

        self.log_dict({
            'train/train_loss': loss,
            'train/train_obs_loss': loss_o,
            'train/train_action_loss': loss_a
            },  
        sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        observations, actions, goal, valid_length, padding_mask,_ = batch
        batch_size, _, _ = observations.shape
        ar_mask_ratio = np.random.choice(self.ar_mask_ratios, 1, p=self.ar_mask_ratio_weights)[0]
        rnd_mask_ratio = np.random.choice(self.rnd_mask_ratios, 1)[0]
        # For convenience, make sure the number of unmasked (obs/action) is the same across examples when masking
        if self.mask_type == 'mae_all':
            observations = observations[:, :self.traj_len]
            actions = actions[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            obs_mask = self.rnd_mask(batch_size, obs_length, rnd_mask_ratio, observations.device)
            action_mask = self.rnd_mask(batch_size, obs_length, rnd_mask_ratio, actions.device)
        elif self.mask_type == 'ae_h':  # ae_h cannot be used for zero-shot
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            # padding history actions as the same length of history obs
            actions = torch.cat([actions[:, :keep_len - 1], torch.zeros(batch_size, 1, actions.shape[-1], device=actions.device)], dim=1)
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.ar_mask(batch_size, keep_len, keep_len, observations.device)
            action_mask = self.ar_mask(batch_size, keep_len, keep_len, actions.device)
        elif self.mask_type == 'mae_h':
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            # padding history actions as the same length of history obs
            actions = torch.cat([actions[:, :keep_len - 1], torch.zeros(batch_size, 1, actions.shape[-1], device=actions.device)], dim=1)
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
            action_mask = self.rnd_mask(batch_size, keep_len - 1, rnd_mask_ratio, actions.device)
            action_mask = torch.cat([action_mask, torch.zeros(batch_size, 1, device=action_mask.device)], dim=1)  # doesn't hope to infer the padding token
        elif self.mask_type == 'mae_f':
            observations = observations[:, :self.traj_len]
            actions = actions[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len =self.ctx_size  # match the beginning of eval (obs length is less than ctx size)
            obs_mask = self.ar_mask(batch_size, obs_length, keep_len, observations.device)
            action_mask = self.ar_mask(batch_size, obs_length, keep_len - 1, observations.device)
        elif self.mask_type == 'mae_rc':
            observations = observations[:, :self.traj_len]
            actions = actions[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))
            history_obs_rnd_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
            history_action_rnd_mask = self.rnd_mask(batch_size, keep_len - 1, rnd_mask_ratio, actions.device)
            history_action_rnd_mask = torch.cat([history_action_rnd_mask, torch.ones(batch_size, 1, device=history_action_rnd_mask.device)], dim=1)
            future_causal_mask = self.ar_mask(batch_size, obs_length - keep_len, 0, observations.device)
            obs_mask = torch.cat([history_obs_rnd_mask, future_causal_mask], dim=1)
            action_mask = torch.cat([history_action_rnd_mask, future_causal_mask], dim=1)
        else:
            raise NotImplementedError
        
        pred_o, pred_a = self(observations, actions, goal, obs_mask, action_mask)
        loss_o, loss_a = self.loss(observations, actions, pred_o, pred_a, padding_mask)

        loss = loss_o + loss_a

        self.log_dict({
            'val/val_loss': loss,
            'val/val_obs_loss': loss_o,
            'val/val_action_loss': loss_a
            },  
        sync_dist=True)
    
    #def on_validation_epoch_end(self):
    #    return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
        }
    
    '''
    # eval funcs
    def init_eval(self, ini_obs: Tensor, goal: Tensor, obs_dim: int, action_dim: int, goal_dim: int):
        # ini_obs: (obs_dim, )
        observations = torch.zeros(size=(1, self.ctx_size, obs_dim), device=self.device)  # 1 x ctx_size x obs_dim
        observations[0, 0] = ini_obs
        # goal: (goal_dim, )
        goal = goal.view(1, -1, goal_dim)
        actions = torch.zeros(size=(1, self.ctx_size, action_dim), device=self.device)  # 1 x ctx_size x action_dim
        return observations, actions, goal

    def make_ar_mask(self, timestep: int):
        
        #make observation mask for first autoregressive ctx_size steps
        
        obs_mask = torch.zeros(size=(1, self.ctx_size), device=self.device)
        action_mask = torch.zeros(size=(1, self.ctx_size), device=self.device)
        action_mask[:, -1] = 1  # the last action is always masked
        if timestep < self.ctx_size - 1:
            obs_mask[0, 1 + timestep:] = 1  # at first ctx_size steps, replace idle observations with obs masked token
            action_mask[0, timestep:] = 1
        return obs_mask, action_mask

    def ar_step(self, timestep: int, observations: Tensor, actions: Tensor, goal: Tensor):
        obs_mask, action_mask = self.make_ar_mask(timestep)
        pred_o, pred_a = self(observations, actions, goal, obs_mask, action_mask)
        if timestep < self.ctx_size - 1:
            action = pred_a[0, timestep]
        else:
            action = pred_a[0, -1]
        return action

    def ar_step_end(self, timestep: int, next_obs: Tensor, action: Tensor, obs_seq: Tensor, action_seq: Tensor):
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
    '''