# PCDT

This is the official Pytorch implementation for the paper:

**Predictive Coding for Decision Transformer**

Tung M. Luu*, Donghoon Lee*, Chang D. Yoo

*equal contribution

A link to our paper can be found on : https://arxiv.org/abs/2410.03408

## Installation


To install requirements:
```
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
pip install git+https://github.com/tunglm2203/pcdt
conda create -n pcdt python=3.7
codna activate pcdt
pip install -r requirements.txt
pip install -e .
```

## Data preprocessing:
```
python -m pcdt.data.parse_d4rl
```


## First Stage : Predictive Coding Learning

Antmaze
```
./scripts\PCDT\launch_pcdt_trl_antmaze_stategoal.sh
```

Kitchen
```
./scripts\PCDT\launch_pcdt_trl_kitchen_stategoal.sh
```

## Second Stage : Policy Learning (DT) based on Predictive Coding

Antmaze
```
./scripts\PCDT\launch_DT_antmaze.sh
```

Kitchen
```
./scripts\PCDT\launch_DT_kitchen.sh
```

## The code will be updated further in the near future.

## Citation
If you use this repo in your research, please consider citing the paper as follows
```
@article{luu2024predictive,
  title={Predictive Coding for Decision Transformer},
  author={Luu, Tung M and Lee, Donghoon and Yoo, Chang D},
  journal={arXiv preprint arXiv:2410.03408},
  year={2024}
}
```

## Acknowledgements
This code is based on top of :
[Decision Transformer](https://github.com/kzl/decision-transformer) , 
[Goal-Conditioned Predictive Coding for Offline Reinforcement Learning](https://github.com/brown-palm/GCPC) , 
[Masked Trajectory Models](https://github.com/facebookresearch/mtm).