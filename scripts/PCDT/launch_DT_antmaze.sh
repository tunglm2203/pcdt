#!/bin/bash

#ENV="antmaze-umaze-v2"
#tjn_ckpt_dir='pcdt_logs/antmaze-umaze-v2'


#ENV="antmaze-umaze-diverse-v2"
#tjn_ckpt_dir='pcdt_logs/antmaze-umaze-diverse-v2/'


#ENV="antmaze-medium-play-v2"
#tjn_ckpt_dir='pcdt_logs/antmaze-medium-play-v2'


#ENV="antmaze-medium-diverse-v2"
#tjn_ckpt_dir='pcdt_logs/antmaze-medium-diverse-v2'


ENV="antmaze-large-play-v2"
#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:10-fh:100-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.19741.ckpt'
#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:100-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.19732.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:120-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.23342.ckpt'
#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:150-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.27752.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:10-fh:70-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.12797.ckpt'
#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:70-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.12818.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:10-fh:85-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.16635.ckpt'
#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:85-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.17005.ckpt'

tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:180-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.31469.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:200-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.33941.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:80-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.14912.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:60-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.07900.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:50-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:1.03459.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:30-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:0.86850.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-play-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:10-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-play-v2-epoch:79-val_loss:0.48616.ckpt'

#ENV="antmaze-large-diverse-v2"
#tjn_ckpt_dir='pcdt_logs/antmaze-large-diverse-v2/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:10-fh:100-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_padzero/checkpoints/antmaze-large-diverse-v2-epoch:79-val_loss:1.10938.ckpt'
#tjn_ckpt_dir='pcdt_logs/antmaze-large-diverse-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:100-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-diverse-v2-epoch:79-val_loss:1.10727.ckpt'

#tjn_ckpt_dir='pcdt_logs/antmaze-large-diverse-v2/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:10-fh:70-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_padzero/checkpoints/antmaze-large-diverse-v2-epoch:79-val_loss:1.04143.ckpt'
#tjn_ckpt_dir='pcdt_logs/antmaze-large-diverse-v2/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:10-fh:70-lr:0.0001-bs:1024-epoch:80-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:10-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/antmaze-large-diverse-v2-epoch:79-val_loss:1.04261.ckpt'

PDROP=0.1
N_LAYERS=4
FUTURE_HORIZON=180
USE_NEW_STACK_SO='True'
PADDING_ZEROS='False'

#INFO="stack_so_new_padzero_fh${FUTURE_HORIZON}_spe_wo_actcond_pdrop${PDROP}_nlayers${N_LAYERS}"
INFO="stack_so_new_wo_padzero_fh${FUTURE_HORIZON}_spe_wo_actcond_pdrop${PDROP}_nlayers${N_LAYERS}"

declare -a GPU_FOR_SEED=(0 1 2 3 4 5 6 7) # List of GPU device id for training with corresponding seeds, length must be same
declare -a SEEDS=(42 43 44 45 46 47 48 49)        # List of seeds for training
#declare -a SEEDS=(0 1 2 3 4 5 6 7)        # List of seeds for training
#declare -a SEEDS=(42)        # List of seeds for training

n_seeds=${#SEEDS[@]}
for ((i = 0; i < n_seeds; i++)); do
  HYDRA_FULL_ERROR=1 python -m pcdt.train_DT_hrl env_name=$ENV \
    model='DT' exp='antmaze_latent_dt' \
    exp.num_epochs=80 \
    model.model_config.n_enc_layers=${N_LAYERS} \
    model.model_config.pdrop=${PDROP} \
    model.model_config.tjn_ckpt_path=$tjn_ckpt_dir \
    wandb.group=$INFO wandb.offline=False info=${INFO} \
    devices=[${GPU_FOR_SEED[$i]}] \
    seed=${SEEDS[$i]} \
    model.model_config.pretrain_use_new_stack_so=$USE_NEW_STACK_SO model.model_config.pretrain_padding_zeros=$PADDING_ZEROS \
    model.model_config.action_condition=False \
    model.model_config.use_pe_as_timestep=False \
    eval_last_k=5 \
    debug=False &
    sleep 8
done

