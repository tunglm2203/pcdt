import os
import wandb

os.environ['WANDB_SILENT'] = 'false'

import torch
torch.set_float32_matmul_precision('medium')

import gym
import warnings
gym.logger.set_level(40)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pcdt.data.sequence import D4RLDataModule_StateGoal_v1
from pcdt.data.parse_d4rl import check_env_name, get_goal_type
from pcdt.model.utils import backup_source, setup_wandb, make_ckpt_filename

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../config', config_name='train_DT')
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    assert check_env_name(cfg.env_name)
    goal_type = 'state_goal'

    # Define dataloader for policy training
    dm = D4RLDataModule_StateGoal_v1(goal_type, cfg)
    if hasattr(cfg.model, 'stage'):
        dm.setup(stage=cfg.model.stage)
    else:
        dm.setup('fit')

    if cfg.model.stage == 'pl':
        base_observation = dm.train.epi_buffers['observations'][0, 0]
    else:
        base_observation = None

    goal_dim = dm.get_obs_dim()
    model = hydra.utils.instantiate(cfg.model,
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
        ctx_size=cfg.exp.ctx_size,
        future_horizon=cfg.exp.future_horizon,
        goal_type=goal_type,
        n_videos_to_render=cfg.n_videos_to_render,
    )

    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(working_dir, 'lightning_logs_ablation', cfg.env_name, cfg.info, make_ckpt_filename(cfg))

    if cfg.debug:
        callbacks = wdb_logger = None
    else:
        backup_source(log_dir)
        wandb_cfg = setup_wandb(cfg, log_dir, dm, goal_dim)
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
        callbacks = [progressbar, lr_monitor, checkpoint]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.devices,
        max_epochs=cfg.exp.num_epochs,
        logger=wdb_logger,
        gradient_clip_val=None,
        callbacks=callbacks,
    )

    print(f'\nSeed: {cfg.seed}')
    print(f'CUDA devices: {cfg.devices}')
    print(f'Log Directory: {log_dir}\n')

    trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    wandb.finish()


if __name__ == '__main__':
    main()