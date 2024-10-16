#!/bin/bash

#ENV="kitchen-complete-v0"
#tjn_ckpt_dir='pcdt_logs/kitchen-complete-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:30-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-complete-v0-epoch:19-val_loss:0.03874.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-complete-v0/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:5-fh:30-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/kitchen-complete-v0-epoch:19-val_loss:0.03788.ckpt'

ENV="kitchen-partial-v0"
tjn_ckpt_dir='pcdt_logs/kitchen-partial-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:30-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-partial-v0-epoch:19-val_loss:0.00577.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-partial-v0/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:5-fh:30-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/kitchen-partial-v0-epoch:19-val_loss:0.00589.ckpt'

#ENV="kitchen-mixed-v0"
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:30-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:19-val_loss:0.00520.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_wo_padzero/trajnet_pcdt-seed:1-ctx:5-fh:30-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_wo_padzero/checkpoints/kitchen-mixed-v0-epoch:19-val_loss:0.00527.ckpt'

#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:10-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:18-val_loss:0.00206.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:20-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:18-val_loss:0.00341.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:40-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:19-val_loss:0.00622.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:50-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:19-val_loss:0.00716.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:60-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:19-val_loss:0.00827.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:70-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:18-val_loss:0.00933.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:90-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:19-val_loss:0.01100.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero/trajnet_pcdt-seed:1-ctx:5-fh:110-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_rc-info:stack_so_new_padzero/checkpoints/kitchen-mixed-v0-epoch:19-val_loss:0.01310.ckpt'
#tjn_ckpt_dir='pcdt_logs/kitchen-mixed-v0/stack_so_new_padzero_mae_h/trajnet_pcdt-seed:1-ctx:5-fh:0-lr:0.0001-bs:1024-epoch:20-d_emb:256-pdrop:0.1-n_enc:2-n_dec:1-n_hd:4-slot:5-mt:mae_h-info:stack_so_new_padzero_mae_h/checkpoints/kitchen-mixed-v0-epoch:18-val_loss:0.00058.ckpt'


PDROP=0.1
N_LAYERS=3
FUTURE_HORIZON=0
USE_NEW_STACK_SO='True'
PADDING_ZEROS='True'

#INFO="stack_so_new_padzero_fh${FUTURE_HORIZON}_spe_wo_actcond_pdrop${PDROP}_nlayers${N_LAYERS}"
#INFO="stack_so_new_wo_padzero_fh${FUTURE_HORIZON}_spe_wo_actcond_pdrop${PDROP}_nlayers${N_LAYERS}"

INFO="goal_condition_pdrop${PDROP}_nlayers${N_LAYERS}"

declare -a GPU_FOR_SEED=(0 1 0 1) # List of GPU device id for training with corresponding seeds, length must be same
declare -a SEEDS=(42 43 44 45)        # List of seeds for training
#declare -a SEEDS=(1)        # List of seeds for training

n_seeds=${#SEEDS[@]}
for ((i = 0; i < n_seeds; i++)); do
  HYDRA_FULL_ERROR=1 python -m pcdt.train_DT_hrl env_name=$ENV \
    model='DT' exp='kitchen_latent_dt' \
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
    model.model_config.goal_condition=True \
    eval_last_k=5 \
    debug=False &
    sleep 8
done

