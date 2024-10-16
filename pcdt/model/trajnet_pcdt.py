import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor, device
from typing import Union
import numpy as np
import math
from omegaconf import DictConfig

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

    def forward(self, slots: Tensor, observations: Tensor, goal: Tensor = None) -> Tensor:
        """
        Args:
            [batch_size, seq_len, embedding_dim]
        """

        if goal is not None:  # for encoder
            # slots += self.pe[:, :slots.shape[1]]
            # goal += self.pe[:, slots.shape[1]:slots.shape[1] + goal.shape[1]]
            # observations += self.pe[:, slots.shape[1] + goal.shape[1]: slots.shape[1] + goal.shape[1] + observations.shape[1]]
            goal += self.pe[:, :goal.shape[1]]
            slots += self.pe[:, goal.shape[1]:goal.shape[1] + slots.shape[1]]
            observations += self.pe[:,
                            slots.shape[1] + goal.shape[1]: slots.shape[1] + goal.shape[1] + observations.shape[1]]
            return slots, goal, observations
        else:  # for decoder
            slots += self.pe[:, :slots.shape[1]]
            observations += self.pe[:, slots.shape[1]: slots.shape[1] + observations.shape[1]]
            return slots, observations



class SlotMAE_PCDT(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, goal_dim: int, ctx_size: int, config: DictConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.ctx_size = ctx_size
        self.embed_dim = config.embed_dim
        self.n_slots = config.n_slots
        self.use_goal = config.use_goal

        self.positional_encoding = SlotMAEPE(d_model=self.embed_dim)
        self.slots = nn.Embedding(self.n_slots, self.embed_dim)
        assert self.n_slots == self.ctx_size

        # encoder
        self.obs_embed = nn.Linear(self.obs_dim, self.embed_dim)
        self.goal_embed = nn.Linear(self.goal_dim, self.embed_dim)

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
        self.obs_decode_emb = nn.Linear(self.embed_dim, self.embed_dim)

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

        self.use_new_stack_so = config.use_new_stack_so
        self.padding_zeros = config.padding_zeros
        if self.use_new_stack_so:
            max_len = 5000
            d_model = self.embed_dim
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def encode(self, observations: Tensor, goal: Tensor, obs_mask: Tensor):
        batch_size, length, _ = observations.shape
        slots = self.slots(torch.arange(self.n_slots, device=observations.device)).repeat(batch_size, 1, 1)

        # goal: B X 1 x goal_dim
        goal_embeddings = self.goal_embed(goal)
        # obs: B X L x obs_dim
        obs_embeddings = self.obs_embed(observations)

        if not self.use_new_stack_so:
            if self.use_goal:
                s, g, o = self.positional_encoding(slots, obs_embeddings, goal_embeddings)
            else:
                s, o = self.positional_encoding(slots, obs_embeddings)
                g = torch.zeros(batch_size, 0, self.embed_dim, device=o.device)

            ##########################################################################################
            # NOTE: Old pretraining scheme
            o_keep = o[obs_mask == 0].view(batch_size, -1, self.embed_dim)
            if s.shape[1] - o_keep.shape[1] != 0:
                o_keep = torch.cat([o_keep, torch.zeros((o_keep.shape[0], s.shape[1] - o_keep.shape[1], self.embed_dim), device='cuda')], dim=1)
            else:
                o_keep = o_keep
            s_o_stack = torch.stack((s, o_keep), dim=1).permute(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
            ##########################################################################################
        else:
            ##########################################################################################
            past_obs = obs_embeddings[:, :self.ctx_size, :]
            g = goal_embeddings

            # NOTE: New pretraining scheme

            s_o_stack = torch.stack((slots, past_obs), dim=1).permute(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)

            mask_for_past = torch.stack((
                torch.zeros_like(obs_mask[:, :self.ctx_size]),
                obs_mask[:, :self.ctx_size]
            ), dim=1).permute(0, 2, 1).reshape(batch_size, -1)

            if self.padding_zeros:
                s_o_stack[mask_for_past == 1] = s_o_stack[mask_for_past == 1] * 0  # Fill the masked token with zeros
                g += self.pe[:, :goal.shape[1]]
                s_o_stack += self.pe[:, goal.shape[1]:goal.shape[1] + s_o_stack.shape[1]]
            else:
                idxes_keep = (mask_for_past == 0).nonzero(as_tuple=True)[1].view(batch_size, -1)
                idxes_masked = (mask_for_past == 1).nonzero(as_tuple=True)[1].view(batch_size, -1)
                idxes_restore = torch.argsort(torch.cat([idxes_keep, idxes_masked], dim=1), dim=1)
                idxes_restore = idxes_restore.unsqueeze(2).repeat(1, 1, self.embed_dim)
                g += self.pe[:, :goal.shape[1]]
                s_o_stack += self.pe[:, goal.shape[1]:goal.shape[1] + s_o_stack.shape[1]]
                s_o_stack = s_o_stack[mask_for_past == 0].view(batch_size, -1, self.embed_dim)       # Keep


        enc_inputs = torch.cat([g, s_o_stack], dim=1)
        encoded_keep = self.encoder(enc_inputs)
        encoded_keep = self.encoder_norm(encoded_keep)

        if not self.use_new_stack_so:
            # encoded_goal = encoded_keep[:, :g.shape[1]]
            bottleneck = encoded_keep[:, g.shape[1]::2, :]
            # encoded_obs = encoded_keep[:, g.shape[1] + 1::2, :]
        else:
            if self.padding_zeros:
                bottleneck = encoded_keep[:, g.shape[1]::2, :]
            else:
                mask_tokens_tmp = torch.zeros((batch_size, idxes_restore.shape[1] - (encoded_keep.shape[1] - goal.shape[1]), self.embed_dim), device=observations.device)
                encoded_keep = torch.cat([encoded_keep[:, g.shape[1]:, :], mask_tokens_tmp], dim=1)
                encoded_keep = torch.gather(encoded_keep, dim=1, index=idxes_restore)
                bottleneck = encoded_keep[:, 0::2, :]

        return bottleneck, None, None
    
    def decode(self, bottleneck: Tensor, obs_mask: Tensor):
        batch_size = bottleneck.shape[0]
        mask_obs = self.obs_mask_token.repeat(obs_mask.shape[0], obs_mask.shape[1], 1)

        b, o = self.positional_encoding(bottleneck, mask_obs)

        dec_inputs = torch.cat([
            torch.stack((b, o[:, :self.ctx_size, :]), dim=1).permute(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim),
            o[:, self.ctx_size:, :]
        ], dim=1)

        decode_out = self.decoder(dec_inputs)
        decode_out = self.decoder_norm(decode_out)

        obs_out = torch.cat([decode_out[:, 1:self.ctx_size * 2:2, :], decode_out[:, self.ctx_size * 2:, :]], dim=1)

        pred_o = self.obs_decoder(obs_out)

        return pred_o

    def forward(self, observations: Tensor, goal: Tensor, obs_mask: Tensor):
        '''
        obs_mask: boolean tensor, True means masked
        '''
        bottleneck, _, _ = self.encode(observations, goal, obs_mask)
        pred_o = self.decode(bottleneck, obs_mask)

        return pred_o
    

class TrajNetPCDT(pl.LightningModule):
    def __init__(self, seed: int, env_name: str, obs_dim: int, action_dim: int, goal_dim: int, lr: float, num_epochs: int, ctx_size: int, future_horizon: int, stage: str, goal_type: str, model_config: DictConfig, **kwargs):
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
        self.model = SlotMAE_PCDT(obs_dim, action_dim, goal_dim, ctx_size, model_config)
        
        self.save_hyperparameters()
    
    def forward(self, observations: Tensor, goal: Tensor, obs_mask: Tensor):
        return self.model.forward(observations, goal, obs_mask)

    def loss(self, target_o: Tensor, pred_o: Tensor, padding_mask: Tensor):
        B, T = padding_mask.size()
        padding_mask = padding_mask.float()

        if self.goal_type == 'state' or self.goal_type == 'state_goal':
            pred_o = pred_o[:, :, :self.model.goal_dim]
            target_o = target_o[:, :, :self.model.goal_dim]

        loss_o = F.mse_loss(pred_o, target_o, reduction='none')
        loss_o *= 1 - padding_mask.view(B, T, -1)  # padding mask: 0 keep, 1 pad, so 1 - padding_mask needed
        loss_o = loss_o.mean() * (B * T) / (1 - padding_mask).sum()

        return loss_o
    
    def ar_mask(self, batch_size: int, length: int, keep_len: float, device: Device):
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        return mask

    def rnd_mask(self, batch_size: int, length: int, mask_ratio: float, device: Device):
        keep_len = max(1, int(np.ceil(length * (1 - mask_ratio))))  # at least keep the first obs

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
        observations, actions, goal, valid_length, padding_mask = batch
        batch_size, _, _ = observations.shape
        ar_mask_ratio = np.random.choice(self.ar_mask_ratios, 1, p=self.ar_mask_ratio_weights)[0]
        rnd_mask_ratio = np.random.choice(self.rnd_mask_ratios, 1)[0]

        # For convenience, make sure the number of unmasked (obs/action) is the same across examples when masking
        if self.mask_type == 'mae_all':
            keep_len = 1
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            obs_mask = self.rnd_mask(batch_size, obs_length, rnd_mask_ratio, observations.device)
        elif self.mask_type == 'ae_h':
            keep_len = max(1, int(np.ceil(self.ctx_size * (1 - ar_mask_ratio))))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.ar_mask(batch_size, keep_len, keep_len, observations.device)
        elif self.mask_type == 'mae_h':
            # keep_len = max(1, int(np.ceil(self.ctx_size * (1 - ar_mask_ratio))))  # match the beginning of eval (obs length is less than ctx size)
            keep_len = self.ctx_size
            observations = observations[:, :keep_len]
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
        elif self.mask_type == 'mae_f':
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(np.ceil(self.ctx_size * (1 - ar_mask_ratio))))  # match the beginning of eval (obs length is less than ctx size)
            obs_mask = self.ar_mask(batch_size, obs_length, keep_len, observations.device)
        elif self.mask_type == 'mae_rc':
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(np.ceil(self.ctx_size * (1 - ar_mask_ratio))))
            history_rnd_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
            future_causal_mask = self.ar_mask(batch_size, obs_length - keep_len, 0, observations.device)
            obs_mask = torch.cat([history_rnd_mask, future_causal_mask], dim=1)
        else:
            raise NotImplementedError
        
        if self.goal_type == 'rtg':
            goal = goal[:, keep_len - 1: keep_len]
        
        pred_o = self(observations, goal, obs_mask)
        loss = self.loss(observations, pred_o, padding_mask)


        self.log_dict({
            'train/train_loss': loss,
            },  
        sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        observations, actions, goal, valid_length, padding_mask = batch
        batch_size, _, _ = observations.shape
        ar_mask_ratio = np.random.choice(self.ar_mask_ratios, 1, p=self.ar_mask_ratio_weights)[0]
        rnd_mask_ratio = np.random.choice(self.rnd_mask_ratios, 1)[0]

        # For convenience, make sure the number of unmasked (obs/action) is the same across examples when masking
        if self.mask_type == 'mae_all':
            keep_len = 1
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            obs_mask = self.rnd_mask(batch_size, obs_length, rnd_mask_ratio, observations.device)
        elif self.mask_type == 'ae_h':
            keep_len = max(1, int(np.ceil(self.ctx_size * (1 - ar_mask_ratio))))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.ar_mask(batch_size, keep_len, keep_len, observations.device)
        elif self.mask_type == 'mae_h':
            # keep_len = max(1, int(np.ceil(self.ctx_size * (1 - ar_mask_ratio))))  # match the beginning of eval (obs length is less than ctx size)
            keep_len = self.ctx_size
            observations = observations[:, :keep_len]
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
        elif self.mask_type == 'mae_f':
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(np.ceil(self.ctx_size * (1 - ar_mask_ratio))))  # match the beginning of eval (obs length is less than ctx size)
            obs_mask = self.ar_mask(batch_size, obs_length, keep_len, observations.device)
        elif self.mask_type == 'mae_rc':
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(np.ceil(self.ctx_size * (1 - ar_mask_ratio))))
            history_rnd_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
            future_causal_mask = self.ar_mask(batch_size, obs_length - keep_len, 0, observations.device)
            obs_mask = torch.cat([history_rnd_mask, future_causal_mask], dim=1)
        else:
            raise NotImplementedError
        
        if self.goal_type == 'rtg':
            goal = goal[:, keep_len - 1: keep_len]
        
        pred_o = self(observations, goal, obs_mask)
        loss = self.loss(observations, pred_o, padding_mask)


        self.log_dict({
            'val/val_loss': loss,
            },  
        sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
        }