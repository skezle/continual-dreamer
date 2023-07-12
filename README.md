# On The Effectiveness of World Models For Continual Reinforcement Learning

This repository contains code to reproduce the experiments in our 2023 Collas paper in which we show that world-models - in particular [DreamerV2](https://github.com/danijar/dreamerv2) - are effective at retaining skills during continual reinforcement learning

```
@article{
}
```

## Using the Package

Create a conda package iwht `python==3.8` and the following packages and versions in `requirements.txt`.

To install `nle` on a machine without root access these [steps](https://github.com/facebookresearch/nle/issues/246) are helpful.

## Minigrid

To install `gym-minigrid`:

```sh 
cd gym_minigrid
pip install -e .
cd ..
```

To run dreamerv2 and dreamerv2 + p2e:

```sh
# DV2
python train_minigrid.py --cl --num_tasks=3 --tag=mg_new_cl_s1_1M --steps=750000 --seed=1 --logdir=logs_cl --del_exp_replay --sep_exp_eval_policies --wandb_proj_name=minigrid_new --minlen=5 --rssm_full_recon
```

```sh
# DV2 + state_bonus
python train_minigrid.py --cl --num_tasks=3 --tag=mg_sb_fr_cl_s0_1M --wandb_proj_name=minigrid_state_bonus --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --sep_exp_eval_policies --wandb_proj_name=minigrid_new --minlen=5 --state_bonus --rssm_full_recon
```

```sh
# DV2 + p2e
#     + random sampling of expl replay buffer
#     + grad heads for obs only
#     + same expl and eval policies
python train_minigrid.py --wandb_proj_name=minigrid_new --cl --num_tasks=3 --tag=mg_new_cl_p2e0.9_s6_1M --steps=750000 --seed=6 --plan2explore --expl_intr_scale=0.9 --expl_extr_scale=0.9 --logdir=logs --del_exp_replay --minlen=50 --rssm_full_recon --sep_exp_eval_policies
```

## Training on Minihack

To run dreamerv2 on cl-small:

```sh
# DV2 
python train_minihack.py --cl --cl_small --num_tasks=4 --tag=mh_cl_small_s0_1M --wandb_proj_name=minihack_task_dist --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --sep_exp_eval_policies --rssm_full_recon --minlen=5 --replay_capacity=1000000
```

To run dreamerv2 + p2e on cl-small:

```sh
# DV2  + p2e
python train_minihack.py --cl --cl_small --num_tasks=4 --tag=mh_cl_small_s0_1M --wandb_proj_name=minihack --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --minlen=5 --replay_capacity=1000000  --plan2explore --expl_intr_scale=0.9 --expl_extr_scale=0.9
```

To run dreamerv2 and rs aka continual-dreamer on cl-small:

```sh
# DV2 + rs 
python train_minihack.py --cl --cl_small --num_tasks=4 --tag=mh_cl_small_rs_s0_1M --wandb_proj_name=minihack_task_dist --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --sep_exp_eval_policies --minlen=5 --replay_capacity=1000000 --reservoir_sampling
```

To run dreamerv2 + p2e + rs aka continual dreamer on cl-small:

```sh
# DV2  + p2e + rs
python train_minihack.py --cl --cl_small --num_tasks=4 --tag=mh_cl_small_rs_s0_1M --wandb_proj_name=minihack --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --minlen=5 --replay_capacity=1000000  --plan2explore --expl_intr_scale=0.9 --expl_extr_scale=0.9 --reservoir_sampling
```

To run dreamerv2 + reservoir sampling + 50:50 on cl-small:

```sh
# DV2 + reservoir sampling + 50:50
python train_minihack.py --cl --cl_small --num_tasks=4 --tag=mh_cl_small_rs_5050_s0_1M --wandb_proj_name=minihack --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --sep_exp_eval_policies --minlen=5 --replay_capacity=1000000 --reservoir_sampling --recent_past_sampl_thres=0.5 
```

To run dreamerv2 + coverage maximization on cl-small:

```sh
# DV2 + coverage maximization
python train_minihack.py --cl --cl_small --num_tasks=4 --tag=mh_cl_small_cm_s0_1M --wandb_proj_name=minihack_task_dist --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --sep_exp_eval_policies --minlen=5 --replay_capacity=1000000 --coverage_sampling
```

To run dreamerv2 + reward sampling on cl-small:

```sh
python train_minihack.py --cl --cl_small --num_tasks=4 --tag=mh_cl_small_rwd_new_s0_1M --wandb_proj_name=minihack_task_dist --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --sep_exp_eval_policies --minlen=5 --replay_capacity=1000000 --reward_sampling
```

### 8 task Minihack

```sh
# DV2
python train_minihack.py --cl --num_tasks=8 --tag=mh_cl_s0_1M --wandb_proj_name=cl_8_tasks_RS --wandb_group=dv2 --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --sep_exp_eval_policies --minlen=5 --replay_capacity=2000000
```

```sh
# DV2 + p2e
python train_minihack.py --cl --num_tasks=8 --tag=mh_cl_p2e_s0_1M --wandb_proj_name=cl_8_tasks_RS --steps=1000000 --seed=0 --logdir=logs_cl --del_exp_replay --minlen=5 --replay_capacity=2000000 --reservoir_sampling
```
