import os
import shutil
import wandb
import gym
from gym_minigrid.wrappers import *
import dreamerv2.api as dv2
from input_args import parse_minigrid_args

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def run_minigrid(args):
    tag = args.tag + "_" + str(args.seed)
    config = dv2.defaults.update({
        'logdir': '{0}/minigrid_{1}'.format(args.logdir, tag),
        'log_every': 1e3,
        'log_every_video': 2e5,
        'train_every': 10,
        'prefill': 1e4,
        'time_limit': 100,  
        'actor_ent': 3e-3,
        'loss_scales.kl': 1.0,
        'discount': 0.99,
        'steps': args.steps,
        'cl': args.cl,
        'num_tasks': args.num_tasks,
        'num_task_repeats': args.num_task_repeats,
        'seed': args.seed,
        'eval_every': 1e4,
        'eval_steps': 1e3,
        'tag': tag,
        "unbalanced_steps": args.unbalanced_steps,
        'replay.capacity': args.replay_capacity,
        'replay.reservoir_sampling': args.reservoir_sampling,
        'replay.recent_past_sampl_thres': args.recent_past_sampl_thres,
        'sep_exp_eval_policies': args.sep_exp_eval_policies,
        'replay.minlen': args.minlen,
        'wandb.group': args.wandb_group,
        'wandb.name': f"{dv2.defaults.expl_behavior}_cl_{tag}" if args.cl else f"{dv2.defaults.expl_behavior}_single-env={args.env}_{tag}", 
        'wandb.project': args.wandb_proj_name,
    }).parse_flags()

    # from https://github.com/danijar/crafter-baselines/blob/main/plan2explore/main.py
    if args.plan2explore:
        config = config.update({
            'expl_behavior': 'Plan2Explore',
            'pred_discount': args.rssm_full_recon,
            'grad_heads': ['decoder', 'reward', 'discount'] if args.rssm_full_recon else ['decoder'],
            'expl_intr_scale': args.expl_intr_scale,
            'expl_extr_scale': args.expl_extr_scale,
            'expl_every': args.expl_every,
            'discount': 0.99,
            'wandb.name':  f"Plan2Explore_cl_{tag}" if args.cl else f"Plan2Explore_single-env={args.env}_{tag}", 
        }).parse_flags()

    wandb.init(
            config=config,
            reinit=True,
            resume=False,
            sync_tensorboard=True,
            dir=args.wandb_dir,
            **config.wandb,
        )

    if config.cl:
        env_names = [
            'MiniGrid-DoorKey-9x9-v0',
            'MiniGrid-LavaCrossingS9N1-v0',
            'MiniGrid-SimpleCrossingS9N1-v0',
        ]


        envs = []
        for i in range(config.num_tasks):
            name = env_names[i]
            env = gym.make(name)
            env = RGBImgPartialObsWrapper(env)  # Get rid of the 'mission' field
            if args.state_bonus:
                assert not args.plan2explore, "state bonus versus plan2explore experiment"
                env = StateBonus(env)
            #env = ReseedWrapper(env, [config.env_seeds[i]])
            envs.append(env)

        if args.eval_skills:
            env_names = [
                'MiniGrid-DoorKey-9x9-v0',
                'MiniGrid-LavaCrossingS9N1-v0',
                'MiniGrid-SimpleCrossingS9N1-v0',
                'MiniGrid-MultiSkill-N2-v0',
            ]

            eval_envs = []
            for i in range(len(env_names)):
                name = env_names[i]
                env = gym.make(name)
                env = RGBImgPartialObsWrapper(env)  # Get rid of the 'mission' field
                # env = ReseedWrapper(env, [config.env_seeds[i]])
                eval_envs.append(env)
        else:
            eval_envs = []
            for i in range(config.num_tasks):
                name = env_names[i]
                env = gym.make(name)
                env = RGBImgPartialObsWrapper(env)  # Get rid of the 'mission' field
                eval_envs.append(env)

        dv2.cl_train_loop(envs, config, eval_envs=eval_envs)

    else:
        env_names = [
            'MiniGrid-DoorKey-9x9-v0',
            'MiniGrid-LavaCrossingS9N1-v0',
            'MiniGrid-SimpleCrossingS9N1-v0',
        ]

        name = env_names[args.env]
        env = gym.make(name)
        env = RGBImgPartialObsWrapper(env)
        dv2.train(env, config)

    if args.del_exp_replay:
        shutil.rmtree(os.path.join(config['logdir'], 'train_episodes'))
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_minigrid_args()
    run_minigrid(args)


