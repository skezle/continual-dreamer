import argparse

def parse_minigrid_args(args=None):
    parser = argparse.ArgumentParser(description="Continual DV2 Minigrid")

    parser.add_argument('--cl', default=False, action='store_true',
                        help='Flag for continual learning loop.')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_task_repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5e5)
    parser.add_argument('--env', type=int, default=0, help='picks the env for the single task dv2.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', type=str, default='', help='unique str to tag tb.')
    parser.add_argument('--del_exp_replay', default=False, action='store_true',
                        help='Flag to delete the training episodes after running the script to save storage space.')
    parser.add_argument('--logdir', type=str, default='logs', help='directory for the tb logs and exp replay episodes.')
    parser.add_argument('--eval_skills', default=False, action='store_true',
                        help='Flag evaluating our model on the multiskill envs.')
    parser.add_argument('--wandb_group', type=str, default='experiment', help='name of the gruop in wandb')
    parser.add_argument('--wandb_proj_name', type=str, default='minihack',
                        help='unique str for wandb projs.')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='unique str for wandb directory.')
    parser.add_argument('--state_bonus', default=False, action='store_true',
                    help='Flag to decide whether to use a state bonus.')
    # DV2
    parser.add_argument('--replay_capacity', type=int, default=2e6)
    parser.add_argument('--reservoir_sampling', default=False, action='store_true',
                        help='Flag for using reservoir sampling.') 
    parser.add_argument('--recent_past_sampl_thres', type=float, default=0.,
                        help="probability of triangle distribution, expected to be > 0 and <= 1. 0 denotes taking episodes always from uniform distribution.")
    parser.add_argument('--minlen', type=int, default=50,
                        help="minimal episode length of episodes stored in the replay buffer")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="mini-batch size")
    parser.add_argument('--unbalanced_steps', type=list, default=None,
                        help="number of steps per each task")

    # exploration
    parser.add_argument('--plan2explore', default=False, action='store_true',
                            help='Enable plan2explore exploration strategy.')
    parser.add_argument('--expl_intr_scale', type=float, default=1.0,
                        help="scale of the intrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_extr_scale', type=float, default=0.0,
                        help="scale of the extrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_every', type=int, default=0, 
                        help="how often to run the exploration strategy.")
    parser.add_argument('--sep_exp_eval_policies', default=False, action='store_true',
                        help='Whether to use separate exploration and evaluation polcies.')
    parser.add_argument('--rssm_full_recon', default=False, action='store_true',
                        help='Whether to have the WM reconstruct the obs, discount and rewards rather than the obs only for p2e only.')

    args = parser.parse_known_args(args=args)[0]
    return args

def parse_minihack_args(args=None):
    parser = argparse.ArgumentParser(description="Continual Dv2")

    parser.add_argument('--cl', default=False, action='store_true',
                        help='Flag for continual learning loop.')
    parser.add_argument('--cl_small', default=False, action='store_true',
                        help='Flag for continual learning loop.')                        
    parser.add_argument('--del_exp_replay', default=False, action='store_true',
                        help='Flag to delete the training episodes after running the script to save storage space.')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_task_repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5e5)
    parser.add_argument('--train_every', type=int, default=10, help="")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env', type=int, default=0, help='picks the env for the single task dv2.')
    parser.add_argument('--tag', type=str, default='', help='unique str to tag tb.')
    parser.add_argument('--logdir', type=str, default='logs', help='directory for the tb logs and exp replay episodes.')
    # Interference
    parser.add_argument("--env_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument('--proc', default=False, action='store_true',
                    help='Flag decide whether to train on 2 interfereing tasks strictly or on procedurally generated envs.')

    # wandb
    parser.add_argument('--wandb_group', type=str, default='experiment', help='name of the gruop in wandb')
    parser.add_argument('--wandb_proj_name', type=str, default='minihack',
                        help='unique str for wandb projs.')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='unique str for wandb directory.')


    # DV2
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=3e-3)
    parser.add_argument('--replay_capacity', type=int, default=2e6)
    parser.add_argument('--rssm_stoch', type=int, default=32,
                        help="number of different stochastic latent variables in the wm")
    parser.add_argument('--rssm_discrete', type=int, default=32,
                        help="number of different classes per stochastic latent variable")
    parser.add_argument('--actor_ent', type=float, default=2e-3,
                        help="entropy coeeficient")
    parser.add_argument('--discount', type=float, default= 0.99,
                        help="discount factor")
    parser.add_argument('--reservoir_sampling', default=False, action='store_true',
                        help='Flag for using reservoir sampling.')  
    parser.add_argument('--uncertainty_sampling', default=False, action='store_true',
                        help='Flag for using uncertainty sampling.')
    parser.add_argument('--recent_past_sampl_thres', type=float, default=0.,
                        help="probability of triangle distribution, expected to be > 0 and <= 1. 0 denotes taking episodes always from uniform distribution.")
    parser.add_argument('--reward_sampling', default=False, action='store_true',
                        help='Flag for using reward sampling.')
    parser.add_argument('--coverage_sampling', default=False, action='store_true',
                        help='Flag for using coverage maximization.')
    parser.add_argument('--coverage_sampling_args', default={"filters": 64, 
                            "kernel_size": [3,3], 
                            "number_of_comparisons": 1000, 
                            "normalize_lstm_state": True, 
                            "distance": "euclid"}, action='store_true',
                        help='Coverage maximization arguments.')
    parser.add_argument('--minlen', type=int, default=50,
                        help="minimal episode length of episodes stored in the replay buffer")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="mini-batch size")
    parser.add_argument('--unbalanced_steps', type=list, default=None,
                        help="number of steps per each task")
    parser.add_argument('--sep_ac', default=False, action='store_true',
                        help='Flag for using separate Actor-Critics per task.')

    # expl
    parser.add_argument('--plan2explore', default=False, action='store_true',
                        help='Enable plan2explore exploration strategy.')
    parser.add_argument('--expl_intr_scale', type=float, default=1.0,
                        help="scale of the intrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_extr_scale', type=float, default=0.0,
                        help="scale of the extrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_every', type=int, default=0, 
                        help="how often to run the exploration strategy.")
    parser.add_argument('--sep_exp_eval_policies', default=False, action='store_true',
                        help='Whether to use separate exploration and evaluation polcies.')
    parser.add_argument('--rssm_full_recon', default=False, action='store_true',
                        help='Whether to have the WM reconstruct the obs, discount and rewards rather than the obs only for p2e only.')
                        
    args = parser.parse_known_args(args=args)[0]
    return args
