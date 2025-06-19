# PCDT

<div><h2>[IROS'24] Predictive Coding for Decision Transformer</h2></div>
<br>

**Tung M. Luu<sup>\*</sup>, Donghoon Lee<sup>\*</sup>, and Chang D. Yoo**
<br>
KAIST, South Korea
<br>
\*Equal contribution
<br>
[[arXiv]](https://arxiv.org/abs/2410.03408) [[Paper]](https://ieeexplore.ieee.org/document/10802437) 


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

## Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@inproceedings{luu2024predictive,
  title={Predictive Coding for Decision Transformer},
  author={Luu, Tung M and Lee, Donghoon and Yoo, Chang D},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgements
- This work was partly supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea
government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by Institute of Information &
communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [RS-2021-II212068, Artificial Intelligence Innovation Hub (Seoul National University).

- This code is based on top of :
[Decision Transformer](https://github.com/kzl/decision-transformer) , 
[Goal-Conditioned Predictive Coding for Offline Reinforcement Learning](https://github.com/brown-palm/GCPC) , 
[Masked Trajectory Models](https://github.com/facebookresearch/mtm).
