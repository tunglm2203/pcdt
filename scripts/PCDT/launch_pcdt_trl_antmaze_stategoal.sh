#!/bin/bash

GPU=0

#ENV="antmaze-umaze-v2"
#ENV="antmaze-umaze-diverse-v2"
#ENV="antmaze-medium-play-v2"
#ENV="antmaze-medium-diverse-v2"
ENV="antmaze-large-play-v2"
#ENV="antmaze-large-diverse-v2"

FUTURE_HORIZON=180

USE_NEW_STACK_SO='True'
PADDING_ZEROS='False'

#INFO="stack_so"
#INFO="stack_so_new_padzero"
INFO="stack_so_new_wo_padzero"
#INFO="no_stack_so"


HYDRA_FULL_ERROR=1 python -m pcdt.train_pcdt_stategoal env_name=$ENV \
  model='trajnet_pcdt' exp=antmaze_trl_goalstate \
  model.model_config.mask_type=mae_rc \
  model.model_config.use_new_stack_so=${USE_NEW_STACK_SO} model.model_config.padding_zeros=${PADDING_ZEROS} \
  exp.num_epochs=80 \
  exp.future_horizon=$FUTURE_HORIZON \
  devices=[$GPU] \
  wandb.group=$INFO wandb.offline=False \
  info=${INFO} \
  debug=False

