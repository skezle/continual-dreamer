import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import gym
import copy
import ast

from dreamerv2.expl import Plan2Explore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
import ruamel.yaml as yaml

import agent
import common

from common import Config
from common import GymWrapper
from common import RenderImage
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput

configs = yaml.safe_load((pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))


def train(env, config, outputs=None):
    # set seeds
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)
    print('GPU available? ', tf.test.is_gpu_available())

    outputs = outputs or [
        common.TerminalOutput(),
        common.JSONLOutput(config.logdir),
        common.TensorBoardOutput(logdir=config.logdir, skipped_metrics=config.skipped_metrics),
    ]
    replay = common.Replay(logdir / 'train_episodes', **config.replay)
    step = common.Counter(replay.stats['total_steps'])
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)
    replay.logger = logger

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video = common.Every(config.log_every_video)
    should_expl = common.Until(config.expl_until)  # config.expl_until == 0 then we are always exploring
    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        logger.scalar('return', score)
        logger.scalar('length', length)
        logger.scalar('task', env.index)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{key}', ep[key].max(0).mean())
        if should_video(step):
            for key in config.log_keys_video:
                logger.video(f'policy_{key}', ep[key])
        logger.add(replay.stats)
        logger.write()

    env = common.GymWrapper(env)
    env = common.ResizeImage(env)
    if hasattr(env.act_space['action'], 'n'):
        env = common.OneHotAction(env)
    else:
        env = common.NormalizeAction(env)
    env = common.TimeLimit(env, config.time_limit)

    driver = common.Driver([env])
    driver.on_episode(per_episode)
    driver.on_step(lambda tran, worker: step.increment())
    driver.on_step(replay.add_step)
    driver.on_reset(replay.add_step)

    prefill = max(0, config.prefill - replay.stats['total_steps'])
    if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(env.act_space)
        driver(random_agent, steps=prefill, episodes=1)
        driver.reset()

    print('Create agent.')
    agnt = agent.Agent(config, env.obs_space, env.act_space, step)

    if isinstance(agnt._expl_behavior, Plan2Explore): #tego nie używamy
        replay.agent = agnt

    dataset = iter(replay.dataset(**config.dataset, oversampling=False))
    dataset_over = iter(replay.dataset(**config.dataset, oversampling=True))

    train_agent = common.CarryOverState(agnt.train)
    data_wm = next(dataset)
    data_pol = next(dataset_over)

    train_agent(data_wm, data_pol)

    if (logdir / 'variables.pkl').exists():
        print("Loading agent.")
        agnt.load(logdir / 'variables.pkl')
    else:
        print('Pretrain agent.')
        for _ in range(config.pretrain):
            data_wm = next(dataset)
            data_pol = next(dataset_over)
            train_agent(data_wm, data_pol)
    policy = lambda *args: agnt.policy(
        *args, mode='explore' if should_expl(step) else 'train')

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                data_wm = next(dataset)
                data_pol = next(dataset_over)
                mets = train_agent(data_wm, data_pol)
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            if should_video(step):
                logger.add(agnt.report(next(dataset)))
            logger.write(fps=True)

    def eval_per_episode(ep):
        task_idx = env.index
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        # print(f'Episode has {length} steps and return {score:.1f}.')
        logger.scalar('eval_return', score)
        logger.scalar('eval_length', length)
        logger.scalar('eval_return_{}'.format(task_idx), score)
        logger.scalar('eval_length_{}'.format(task_idx), length)
        if should_video(step):
            for key in config.log_keys_video:
                logger.video(f'eval_{task_idx}_{step.value}', ep[key])
        logger.write()

    driver.on_step(train_step)

    eval_driver = common.Driver([env])
    eval_driver.on_episode(eval_per_episode)  # cl eval loop
    # in the original api the evaluation policy and the training policy are the same
    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    while step < config.steps:
        logger.write()
        driver(policy, steps=config.eval_every)
        if config.sep_exp_eval_policies:
            eval_driver(eval_policy, steps=config.eval_steps)
        else:
            eval_driver(policy, steps=config.eval_steps)
        agnt.save(logdir / 'variables.pkl')


def cl_train_loop(envs, config, outputs=None, eval_envs=None):
    # set seeds
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    unbalanced_steps = ast.literal_eval(config.unbalanced_steps)

    # TB and configs
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)
    print('GPU available? ', tf.test.is_gpu_available())

    outputs = outputs or [
        common.TerminalOutput(),
        common.JSONLOutput(config.logdir),
        common.TensorBoardOutput(logdir=config.logdir, skipped_metrics=config.skipped_metrics),
    ]

    replay = common.Replay(
        logdir / 'train_episodes', **config.replay, num_tasks=config.num_tasks)

    total_step = common.Counter(replay.stats['total_steps'])
    print("Replay buffer total steps: {}".format(replay.stats['total_steps']))
    logger = common.Logger(total_step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)
    replay.logger = logger

    # from replay buffer we can warm start the CL loop and work out the task we are currently in and the step
    if unbalanced_steps is not None:
        tot_steps_after_task = np.cumsum(unbalanced_steps)
        task_id = next((i for i, j in enumerate(replay.stats['total_steps'] < tot_steps_after_task) if j), None)
        print("Task {}".format(task_id))
        rep = int(replay.stats['total_steps'] // (np.sum(unbalanced_steps)))
        print("Rep {}".format(rep))
        restart_step = (unbalanced_steps - tot_steps_after_task + replay.stats['total_steps'])[task_id]
    else:
        task_id = int(replay.stats['total_steps'] // config.steps)
        print("Task {}".format(task_id))
        rep = int(replay.stats['total_steps'] // (config.steps * config.num_tasks))
        print("Rep {}".format(rep))
        restart_step = int(replay.stats['total_steps'] % config.steps)
    print("Restart step: {}".format(restart_step))
    restart = True if restart_step > 0 else False

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video = common.Every(config.log_every_video)
    should_video_eval = common.Every(config.log_every_video)
    if config.expl_every:
        print("exploring every {} steps".format(config.expl_every))
        should_expl = common.Every(config.expl_every)
    else:
        should_expl = common.Until(config.expl_until)
    should_recon = common.Every(config.log_recon_every)

    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        logger.scalar('return', score)
        logger.scalar('length', length)
        logger.scalar('task', task_id)
        logger.scalar('replay_capacity', replay.stats['loaded_steps'])
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{key}', ep[key].max(0).mean())
        if should_video(total_step):
            for key in config.log_keys_video:
                logger.video(f'policy_{key}', ep[key])
        logger.add(replay.stats)
        logger.write()

    def create_envs_drivers(_env):
        _env = common.GymWrapper(_env)
        _env = common.ResizeImage(_env)
        if hasattr(_env.act_space['action'], 'n'):
            _env = common.OneHotAction(_env)
        else:
            _env = common.NormalizeAction(_env)
        _env = common.TimeLimit(_env, config.time_limit)

        driver = common.Driver([_env])
        driver.on_episode(per_episode)
        driver.on_step(lambda tran, worker: total_step.increment())
        driver.on_step(replay.add_step)
        driver.on_reset(replay.add_step)
        driver.on_step(lambda tran, worker: step.increment())
        return _env, driver

    # for evaluation driver
    if eval_envs is None:
        eval_envs = envs

    _eval_envs = []
    for i in range(len(eval_envs)):
        env, _ = create_envs_drivers(eval_envs[i])
        _eval_envs.append(env)

    while rep < config.num_task_repeats:
        while task_id < len(envs):
            print("\n\t Task {} Rep {} \n".format(task_id + 1, rep + 1))

            env = envs[task_id]
            if restart:
                start_step = restart_step
                restart = False
            else:
                start_step = 0

            replay.set_task(task_id)

            replay.set_task(task_id)

            step = common.Counter(start_step)

            env = common.GymWrapper(env)
            env = common.ResizeImage(env)
            if hasattr(env.act_space['action'], 'n'):
                env = common.OneHotAction(env)
            else:
                env = common.NormalizeAction(env)
            env = common.TimeLimit(env, config.time_limit)

            driver = common.Driver([env])
            driver.on_episode(per_episode)
            driver.on_step(lambda tran, worker: total_step.increment())
            driver.on_step(replay.add_step)
            driver.on_reset(replay.add_step)
            driver.on_step(lambda tran, worker: step.increment())

            # after round 1, task 1 the prefill will not happen
            prefill = max(0, config.prefill - replay.stats['total_steps'])
            if prefill:
                print(f'Prefill dataset ({prefill} steps).')
                random_agent = common.RandomAgent(env.act_space)
                driver(random_agent, steps=prefill, episodes=1)
                driver.reset()

            print('Create agent.')
            agnt = agent.Agent(config, env.obs_space, env.act_space, total_step)

            if isinstance(agnt._expl_behavior, Plan2Explore):
                replay.agent = agnt

            # dataset: {batch: 16, length: 50}
            dataset = iter(replay.dataset(**config.dataset))
            train_agent = common.CarryOverState(agnt.train)
            train_agent(next(dataset))
            if (logdir / 'variables.pkl').exists():
                print("Loading agent.")
                agnt.load(logdir / 'variables.pkl')
            else:
                print('Pretrain agent.')
                for _ in range(config.pretrain):
                    train_agent(next(dataset))

            policy = lambda *args: agnt.policy(
                *args, mode='explore' if should_expl(total_step) else 'train')

            def eval_per_episode(ep, task_idx):
                length = len(ep['reward']) - 1
                score = float(ep['reward'].astype(np.float64).sum())
                # print(f'Episode has {length} steps and return {score:.1f}.')
                logger.scalar('eval_return_{}'.format(task_idx), score)
                logger.scalar('eval_length_{}'.format(task_idx), length)
                if should_video_eval(total_step):
                    for key in config.log_keys_video:
                        logger.video(f'eval_{task_idx}_{total_step.value}', ep[key])
                ep = {k: np.expand_dims(v, axis=0) for k, v in ep.items()}
                if should_recon(total_step):
                    model_loss, _, _, _ = agnt.wm.loss(ep)
                    logger.scalar('eval_recon_loss_{}'.format(task_idx), model_loss)
                logger.write()

            def train_step(tran, worker):
                if should_train(total_step):
                    for _ in range(config.train_steps):
                        mets = train_agent(next(dataset))
                        [metrics[key].append(value) for key, value in mets.items()]
                if should_log(total_step):
                    for name, values in metrics.items():
                        logger.scalar(name, np.array(values, np.float64).mean())
                        metrics[name].clear()
                    if should_video(total_step):
                        logger.add(agnt.report(next(dataset)))
                    logger.write(fps=True)

            driver.on_step(train_step)

            eval_driver = common.Driver(_eval_envs, cl=config.cl)
            eval_driver.on_episode(eval_per_episode)  # cl eval loop
            # in the original api the evaluation policy and the training policy are the same
            eval_policy = lambda *args: agnt.policy(*args, mode='eval')

            # step is incremented on every step of the driver, however the driver
            # will execute for at least config.eval_every number of times, so loop will
            # not end exact once step >= config.steps.
            if unbalanced_steps is not None:
                steps_limit = int(unbalanced_steps[task_id])
            else:
                steps_limit = config.steps

            while step < steps_limit:
                logger.write()
                driver(policy, steps=config.eval_every)
                if config.sep_exp_eval_policies:
                    eval_driver(eval_policy, steps=config.eval_steps)
                else:
                    eval_driver(policy, steps=config.eval_steps)
                agnt.save(logdir / 'variables.pkl')

            # increment the task id
            task_id += 1

        # increment the number of reps
        rep += 1