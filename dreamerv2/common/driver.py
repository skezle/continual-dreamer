import numpy as np


class Driver:

    def __init__(self, envs, cl=False, **kwargs):
        self._envs = envs
        self._cl = cl
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None


    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            obs = {
                i: self._envs[i].reset()
                for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
            for i, ob in obs.items():
                self._obs[i] = ob() if callable(ob) else ob
                act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, task_index=self._envs[0].index) for fn in self._on_resets]
                self._eps[i] = [tran]
            obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
            actions, self._state = policy(obs, self._state, **self._kwargs)
            actions = [
                {k: np.array(actions[k][i]) for k in actions}
                for i in range(len(self._envs))]
            assert len(actions) == len(self._envs)
            obs = []
            for e, a in zip(self._envs, actions):
                try:
                    ob = e.step(a)
                except RuntimeError:
                    # sometimes get a nethack env is closed error, so we are calling a step
                    # once the episode has already terminated and Nethack throws and error.
                    # here: nle/win/rn/pynethack.cc L159
                    # this is a bit of a tempory fix - let's reset the env.

                    # sometimes the minigrid envs will through an error where
                    # the assertion in line 354 of envs.py is hit and thus needs
                    # a reset.
                    ob = e.reset()
                    # import pdb; pdb.set_trace()
                obs.append(ob)
            # obs = [e.step(a) for e, a in zip(self._envs, actions)]
            obs = [ob() if callable(ob) else ob for ob in obs]
            for i, (act, ob) in enumerate(zip(actions, obs)):
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                for j in range(len(self._on_steps)):
                    if j == 1:
                        self._on_steps[j](tran, worker=i, task_index=self._envs[0].index)
                    else:
                        self._on_steps[j](tran, worker=i, **self._kwargs)
                self._eps[i].append(tran)
                step += 1
                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    if self._cl:
                        [fn(ep, task_idx=i, **self._kwargs) for fn in self._on_episodes]
                    else:
                        [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
            self._obs = obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
