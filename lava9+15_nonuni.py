import gym
import gym_minigrid
import dreamerv2.api as dv2 #kafel
from gym_minigrid.envs.crossing import CrossingEnv
import wandb

config = dv2.defaults.update({
    'logdir': 'adasdfggsaffadsasddjflaasllllkglkasadkdsdlfsa', #'/net/tscratch/people/plgtaddziarm/logdir/27.11/lava15_nonuni_over_policy',
    'log_every': 1e3,
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e4,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    'eval_every': 1e4,
    'eval_steps': 1e3,
    'log_every_video': 2e5,
    'time_limit': 100,
}).parse_flags()

#wandb.init(
#            project="dreamer_ovprintersampling_policy_test",
#            name="lava15_nonuni",
#            config=config,
#            reinit=True,
#            resume=False,
#            sync_tensorboard=True,
#        )

env2 = CrossingEnv(size=15, num_crossings=1)
env2 = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env2)
env1 = gym.make('MiniGrid-LavaCrossingS9N1-v0')
env1 = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env1)

list_env = [env1, env2]

import math
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
#from rendering import *

class ConnectedEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

        # Done completing task
        done = 6

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7
    ):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        self.index = 0 #random from 1 to 5, drawn in reset()
        self.counter = 0

        # Action enumeration for this environment
        self.actions = ConnectedEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        if np.random.uniform(0,1) < 0.99:
            self.index = 0
        else:
            self.index = 1

        obs = list_env[self.index].reset()
        print(self.index)
        return obs


    def step(self, action):
        return list_env[self.index].step(action)

envik = ConnectedEnv()
#env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)

dv2.train(envik, config)

#wandb.finish()