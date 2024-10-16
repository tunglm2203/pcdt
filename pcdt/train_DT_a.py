import os
import torch
torch.set_float32_matmul_precision('medium')
import shutil
import datetime
import gym
gym.logger.set_level(40)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from .data.sequence import D4RLDataModule_StateGoal_v0
from .data.parse_d4rl import check_env_name, get_goal_type

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf


def make_ckpt_filename(cfg: DictConfig):
    env_name = cfg.env_name[:-3]  
    ckptfile_name = f'{env_name}-s{cfg.seed}'
    if hasattr(cfg, 'info') and cfg.info != '':
        ckptfile_name += f'-{cfg.info}'
    return ckptfile_name

def setup_wandb(cfg: DictConfig, working_dir, dataloader, goal_dim):
    # Attempt to preprocess all config
    _cfg = cfg.copy()
    _cfg.exp.obs_dim = dataloader.get_obs_dim()
    _cfg.exp.act_dim = dataloader.get_action_dim()
    _cfg.exp.goal_dim = goal_dim
    _cfg = OmegaConf.to_container(_cfg, resolve=True)

    unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_model = False if cfg.wandb.offline else True
    tags = [cfg.wandb.group] if cfg.wandb.tags is None else cfg.wandb.tags
    wandb_cfg = {
        'config': _cfg,
        'name': make_ckpt_filename(cfg), 'project': cfg.wandb.project,
        'group': cfg.wandb.group, 'tags': tags,
        'save_dir': working_dir, 'offline': cfg.wandb.offline, 'entity': cfg.wandb.entity,
        'save_code': True,
        'log_model': log_model, 'id': unique_identifier
    }
    return wandb_cfg


def backup_source(log_dir):
    source_code_backup = os.path.join(log_dir, 'source_codes')
    os.makedirs(log_dir, exist_ok=True)
    shutil.copytree("pcdt", os.path.join(source_code_backup, "pcdt"),
                    ignore=shutil.ignore_patterns('*.pyc', '*.so', '*.cpp', '*.h', '*.pyi', '*.pxd',
                                                  '*.typed', '__pycache__', '*.pkl', '*.npz', '*.pt', '*.ckpt'),
                    )
    

@hydra.main(version_base=None, config_path='../config', config_name='train_DT_a')
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed) # set seed

    assert check_env_name(cfg.env_name)
    # goal_type = get_goal_type(cfg.env_name)
    goal_type = 'state_goal'  # x,y position in antmzae

    # Define dataloader
    dm = D4RLDataModule_StateGoal_v0(goal_type, cfg)
    dm.setup(stage='fit')

    base_observation = dm.train.epi_buffers['observations'][0, 0]

    if goal_type == 'state_goal':
        goal_dim = dm.get_obs_dim()
    else:
        goal_dim = cfg.exp.goal_dim

    # Define model
    
    
    
    model = hydra.utils.instantiate(
        cfg.model,
        seed=cfg.seed,
        env_name=cfg.env_name,
        obs_dim=dm.get_obs_dim(),
        act_dim=dm.get_action_dim(),
        goal_dim=goal_dim,
        lr=cfg.exp.lr,
        num_eval_rollouts=cfg.exp.num_eval_rollouts,
        eval_last_k=cfg.eval_last_k,
        base_observation=base_observation,
        num_epochs=cfg.exp.num_epochs,
        use_scheduler=cfg.exp.use_scheduler,
        goal_type=goal_type,
        ctx_size=cfg.exp.ctx_size,
        future_horizon=cfg.exp.future_horizon
    )
    
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(working_dir, 'lightning_logs', cfg.env_name, cfg.info, make_ckpt_filename(cfg))
    #backup_source(log_dir)
    wandb_cfg = setup_wandb(cfg, working_dir, dm, goal_dim)
    wdb_logger = WandbLogger(**wandb_cfg)

    # Define callbacks
    progressbar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint = ModelCheckpoint(
        monitor='epoch',
        dirpath=os.path.join(working_dir, 'lightning_logs', cfg.env_name, cfg.info, make_ckpt_filename(cfg), 'checkpoints'),
        filename=cfg.env_name + '-epoch_{epoch}-score_{val/norm_score_hrl:.3f}',
        mode='max',
        save_top_k=cfg.eval_last_k,
        auto_insert_metric_name=False
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.devices,
        max_epochs=cfg.exp.num_epochs,
        logger=wdb_logger,
        gradient_clip_val=None,
        callbacks=[progressbar, lr_monitor, checkpoint],
    )

    print(f'\nSeed: {cfg.seed}')
    print(f'CUDA devices: {cfg.devices}')
    print(f'Log Directory: {os.path.join(cfg.env_name, make_ckpt_filename(cfg))}\n')

    trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())


if __name__ == '__main__':
    main()