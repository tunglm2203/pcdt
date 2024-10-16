import os
import torch
torch.set_float32_matmul_precision('medium')

import gym
gym.logger.set_level(40)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pcdt.data.sequence import D4RLDataModule_StateGoal_v0
from pcdt.data.parse_d4rl import check_env_name, get_goal_type
from pcdt.model.utils import setup_wandb, backup_source

import hydra
from omegaconf import DictConfig


def make_ckpt_filename(cfg: DictConfig):
    ckptfile_name = f'{cfg.model._target_.split(".")[2]}-seed:{cfg.seed}-ctx:{cfg.exp.ctx_size}-fh:{cfg.exp.future_horizon}-lr:{cfg.exp.lr}-bs:{cfg.exp.batch_size * len(cfg.devices)}-epoch:{cfg.exp.num_epochs}-d_emb:{cfg.model.model_config.embed_dim}-pdrop:{cfg.model.model_config.pdrop}'
    if hasattr(cfg.model.model_config, 'n_enc_layers'):
        ckptfile_name += f'-n_enc:{cfg.model.model_config.n_enc_layers}'
    if hasattr(cfg.model.model_config, 'n_dec_layers'):
        ckptfile_name += f'-n_dec:{cfg.model.model_config.n_dec_layers}'
    if hasattr(cfg.model.model_config, 'n_head'):
        ckptfile_name += f'-n_hd:{cfg.model.model_config.n_head}'
    if hasattr(cfg.model.model_config, 'n_slots'):
        ckptfile_name += f'-slot:{cfg.model.model_config.n_slots}'
    if hasattr(cfg.model.model_config, 'mask_type'):
        ckptfile_name += f'-mt:{cfg.model.model_config.mask_type}'
    if hasattr(cfg, 'info') and cfg.info != '':
        ckptfile_name += f'-info:{cfg.info}'
    return ckptfile_name


@hydra.main(version_base=None, config_path='../config', config_name='train_stategoal')
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    assert check_env_name(cfg.env_name)
    # goal_type = get_goal_type(cfg.env_name)
    goal_type = 'state_goal'

    dm = D4RLDataModule_StateGoal_v0(goal_type, cfg)
    if hasattr(cfg.model, 'stage'):
        dm.setup(stage=cfg.model.stage)
    else:
        dm.setup('fit')

    if cfg.model.stage == 'pl':
        base_observation = dm.train.epi_buffers['observations'][0, 0]
    else:
        base_observation = None

    goal_dim = dm.get_obs_dim()
    pcdt = hydra.utils.instantiate(cfg.model,
        seed=cfg.seed,
        env_name=cfg.env_name,
        obs_dim=dm.get_obs_dim(),
        action_dim=dm.get_action_dim(),
        goal_dim=goal_dim,
        num_epochs=cfg.exp.num_epochs,
        lr=cfg.exp.lr,
        use_scheduler=cfg.exp.use_scheduler,
        ctx_size=cfg.exp.ctx_size,
        future_horizon=cfg.exp.future_horizon,
        num_eval_rollouts=cfg.exp.num_eval_rollouts,
        eval_last_k=cfg.eval_last_k,
        goal_type=goal_type,
        base_observation=base_observation
    )

    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(working_dir, 'pcdt_logs', cfg.env_name, cfg.info, make_ckpt_filename(cfg))
    wdb_logger = None
    if not cfg.debug:
        backup_source(log_dir)
        wandb_cfg = setup_wandb(cfg, log_dir, dm, goal_dim, name=f'{cfg.env_name}-{make_ckpt_filename(cfg)}')
        wdb_logger = WandbLogger(**wandb_cfg)

    progressbar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint = ModelCheckpoint(
        monitor='epoch',
        dirpath=os.path.join(log_dir, 'checkpoints'),
        filename=cfg.env_name + '-epoch_{epoch}-score_{eval/norm_score:.3f}',
        mode='max',
        save_top_k=cfg.eval_last_k,
        auto_insert_metric_name=False
    )

    if hasattr(cfg.model, 'stage') and cfg.model.stage == 'trl':
        checkpoint = ModelCheckpoint(
            monitor='val/val_loss',
            dirpath=os.path.join(log_dir, 'checkpoints'),
            filename=cfg.env_name + '-epoch:{epoch}-val_loss:{val/val_loss:.5f}',
            mode='min',
            save_top_k=cfg.eval_last_k,
            auto_insert_metric_name=False
        )


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.devices,
        max_epochs=cfg.exp.num_epochs,
        logger=wdb_logger,
        gradient_clip_val=1.0,
        callbacks=[progressbar, lr_monitor, checkpoint] if not cfg.debug else None,
    )

    print(f'\nSeed: {cfg.seed}')
    print(f'CUDA devices: {cfg.devices}')
    print(f'Log Directory: {os.path.join(cfg.env_name, make_ckpt_filename(cfg))}\n')

    trainer.fit(model=pcdt, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())


if __name__ == '__main__':
    main()