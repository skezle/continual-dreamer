import os
import shutil
import wandb
import gym
import minihack
import numpy as np
import dreamerv2.api as dv2
import wandb
from input_args import parse_minihack_args
import ast

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

class MiniHackObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(84, 84, 3))

    def observation(self, obs):
        obs = obs["pixel_crop"]
        obs = np.pad(obs, [(2, 2), (2, 2), (0, 0)])
        return obs

# from https://github.com/MiniHackPlanet/MiniHack/blob/e9c8c20fb2449d1f87163314f9b3617cf4f0e088/minihack/scripts/venv_demo.py#L28
class MiniHackMakeVecSafeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.basedir = os.getcwd()

    def step(self, action: int):
        os.chdir(self.env.env._vardir)
        x = self.env.step(action)
        os.chdir(self.basedir)
        return x

    def reset(self):
        os.chdir(self.env.env._vardir)
        x = self.env.reset()
        os.chdir(self.basedir)
        return x

    def close(self):
        os.chdir(self.env.env._vardir)
        self.env.close()
        os.chdir(self.basedir)

    def seed(self, core=None, disp=None, reseed=False):
        os.chdir(self.env.env._vardir)
        self.env.seed(core, disp, reseed)
        os.chdir(self.basedir)

def make_minihack(
    env_name,
    observation_keys=["pixel_crop", "pixel", "glyphs"],
    reward_win=1,
    reward_lose=0,
    penalty_time=0.0,
    penalty_step=-0.001,  # MiniHack uses different than -0.01 default of NLE
    penalty_mode="constant",
    character="mon-hum-neu-mal",
    savedir=None,
    # save_tty=False -> savedir=None, see https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/common/envs/tasks.py#L168
    **kwargs,
):
    env = gym.make(
        f"MiniHack-{env_name}",
        observation_keys=observation_keys,
        reward_win=reward_win,
        reward_lose=reward_lose,
        penalty_time=penalty_time,
        penalty_step=penalty_step,
        penalty_mode=penalty_mode,
        character=character,
        savedir=savedir,
        **kwargs,
    )  # each env specifies its own self._max_episode_steps
    env = MiniHackMakeVecSafeWrapper(env)
    env = MiniHackObsWrapper(env)
    return env

def run_minihack(args):

    config = dv2.defaults
    config = config.update(dv2.configs['crafter'])
    tag = args.tag + str(args.seed)

    config = config.update({
        'logdir': '{0}/minihack_{1}'.format(args.logdir, tag),
        'log_every': 1e3,
        'log_every_video': 2e5,
        'train_every': args.train_every,
        'time_limit': 100,
        'prefill': 1e4,
        # 'actor_ent': args.eta,
        'loss_scales.kl': args.beta,
        'steps': args.steps,
        "unbalanced_steps": args.unbalanced_steps,
        'cl': args.cl,
        'cl_small': args.cl_small,
        'num_tasks': args.num_tasks,
        'num_task_repeats': args.num_task_repeats,
        'seed': args.seed,
        'eval_every': 5e4,
        'eval_steps': 1e3,
        'tag': tag,
        "dataset.batch": args.batch_size,
        'replay.capacity': args.replay_capacity,
        'replay.reservoir_sampling': args.reservoir_sampling,
        "replay.uncertainty_sampling": args.uncertainty_sampling,
        'replay.recent_past_sampl_thres': args.recent_past_sampl_thres,
        'replay.reward_sampling': args.reward_sampling,
        'replay.coverage_sampling': args.coverage_sampling,
        'replay.coverage_sampling_args': args.coverage_sampling_args,
        'replay.minlen': args.minlen,
        'sep_exp_eval_policies': args.sep_exp_eval_policies,
        "rssm.stoch": args.rssm_stoch,
        "rssm.discrete": args.rssm_discrete,
        "actor_ent": args.actor_ent,
        "discount": args.discount,
        'wandb.group': args.wandb_group,
        'wandb.name': f"{dv2.defaults.expl_behavior}_cl-small={args.cl_small}_{tag}" if args.cl else f"{dv2.defaults.expl_behavior}_single-env={args.env}_{tag}", 
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
            'discount': 0.99,
            'wandb.name': f"Plan2Explore_cl-small={args.cl_small}_{tag}" if args.cl else f"Plan2Explore_single-env={args.env}_{tag}",
        }).parse_flags()
    
    unbalanced_steps = ast.literal_eval(config.unbalanced_steps)
    if config.cl:
        if config.cl_small:
            env_names = [
                "Room-Random-15x15-v0",
                "Room-Trap-15x15-v0",
                "River-Narrow-v0",
                "River-Monster-v0",
            ]
        elif unbalanced_steps is not None:
            env_names = [
                    "Room-Random-15x15-v0",
                    "River-Narrow-v0",
                ]
        else:
            env_names = [
                "Room-Random-15x15-v0",  # |A|=8 consider replacing with "Room-Ultimate-5x5-v0",
                "Room-Monster-15x15-v0",  # |A|=8
                "Room-Trap-15x15-v0",  # |A|=8
                "Room-Ultimate-15x15-v0",  # |A|=8
                "River-Narrow-v0",
                "River-v0",
                "River-Monster-v0",
                "HideNSeek-v0",
            ]

        wandb.init(
            config=config,
            reinit=True,
            resume=False,
            sync_tensorboard=True,
            **config.wandb,
        )

        envs = []
        for i in range(config.num_tasks):
            name = env_names[i]
            env = make_minihack(name)
            print("env {0}, action space: {1}".format(name, env.action_space.n))
            envs.append(env)

        dv2.cl_train_loop(envs, config)
    else:
        envs = [
            "Room-Random-15x15-v0",
            "Room-Monster-15x15-v0",
            "Room-Trap-15x15-v0",
            "Room-Ultimate-15x15-v0",
            "River-Narrow-v0",
            "River-v0",
            "River-Monster-v0",
            "HideNSeek-v0",
            "CorridorBattle-v0",
            "River-Lava-v0",
            "River-MonsterLava-v0",
        ]

        config = config.update({
            'tag': tag + envs[args.env],
        }).parse_flags()

        wandb.init(
            config=config,
            reinit=True,
            resume=False,
            sync_tensorboard=True,
            **config.wandb,
        )

        env = make_minihack(envs[args.env])
        dv2.train(env, config)

    if args.del_exp_replay:
        shutil.rmtree(os.path.join(config['logdir'], 'train_episodes'))

    wandb.finish()

if __name__ == "__main__":
    args = parse_minihack_args()
    run_minihack(args)
