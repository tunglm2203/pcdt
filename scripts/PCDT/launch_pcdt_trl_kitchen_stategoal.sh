#!/bin/bash

GPU=0

#ENV="kitchen-complete-v0"
ENV="kitchen-mixed-v0"
#ENV="kitchen-partial-v0"




FUTURE_HORIZON=0

USE_NEW_STACK_SO='True'
PADDING_ZEROS='True'

#INFO="stack_so"
INFO="stack_so_new_padzero_mae_h"
#INFO="stack_so_new_wo_padzero"
#INFO="no_stack_so"


HYDRA_FULL_ERROR=1 python -m pcdt.train_pcdt_stategoal env_name=$ENV \
  model='trajnet_pcdt' exp=kitchen_trl_goalstate \
  model.model_config.mask_type=mae_rc \
  model.model_config.use_new_stack_so=${USE_NEW_STACK_SO} model.model_config.padding_zeros=${PADDING_ZEROS} \
  model.model_config.mask_type='mae_h' \
  exp.num_epochs=20 \
  exp.future_horizon=$FUTURE_HORIZON \
  devices=[$GPU] \
  wandb.group=$INFO wandb.offline=False \
  info=${INFO} \
  debug=False