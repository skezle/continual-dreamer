from typing import Dict, List
import numpy as np
import pandas as pd

def smooth(
    scalars: list,
    weight: float, # Weight between 0 and 1
) -> list:  
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def performance(
    data: Dict[int, np.array],
    num_tasks: int=8,
    num_steps: int=1e6,
    num_seeds: int=10,
    smoothing_factor: bool=None,
    verbose: bool=False,
) -> np.array:
    """
    Av performance

    N = num_tasks
    f = \sum_i=1^N p_i
    p_i = perf(num_steps*num_tasks)
    """
    keys = list(data.keys())
    # forgetting
    p_T = np.zeros(num_seeds) # num_seeds
    for i in range(num_tasks):
        # performance data per minihack task
        k = keys[i]
        d = data[k]
        # dataframe construction for different numbers of seeds
        df = {'t': d[:, 0]}
        for j in range(d.shape[1] - 1):
            df['s{}'.format(j)] = d[:, j + 1]
        dataset = pd.DataFrame(df).fillna(method='ffill').fillna(method='bfill')

        # final performance on minihack task i
        if smoothing_factor is None:
            p_i_T = dataset.tail(1).to_numpy()[0, 1:]
        else:
            for j in range(d.shape[1] - 1):
                dataset['s{}'.format(j)] = smooth(dataset['s{}'.format(j)].to_numpy(), smoothing_factor)
            p_i_T = dataset.tail(1).to_numpy()[0, 1:]

        # minihack levels can be negative so let's scale up to 0
        p_i_T[p_i_T <= 0] = 0
        p_T += p_i_T
        if verbose:
            print("Task {0}, Performance {1}".format(i + 1, p_i_T))
    return p_T / num_tasks

def forgetting(
    data: Dict[int, np.array],
    num_tasks: int=8,
    num_steps: int=1e6,
    num_seeds: int=10,
    smoothing_factor: float=None,
    verbose: bool=False,
) -> np.array:
    """
    N = num_tasks
    f = \sum_i=1^N f_i
    f_i = perf(i*num_steps) - perf(num_steps*num_tasks)
    """
    keys = list(data.keys())
    # forgetting
    f = np.zeros(num_seeds)  # num_seeds
    for i in range(num_tasks):
        # performance data per minihack task
        key = keys[i]
        d = data[key]
        # dataframe construction for different numbers of seeds
        df = {'t': d[:, 0]}
        for j in range(d.shape[1] - 1):
            df['s{}'.format(j)] = d[:, j + 1]
        dataset = pd.DataFrame(df).fillna(method='ffill').fillna(method='bfill')

        if smoothing_factor is None:
            f_i = dataset[dataset['t'] <= ((i + 1) * num_steps)].tail(1).to_numpy()[0, 1:]
        else:
            for j in range(num_seeds):
                dataset['s{}'.format(j)] = smooth(dataset['s{}'.format(j)].to_numpy(), smoothing_factor)
        
            f_i = dataset[dataset['t'] <= ((i + 1) * num_steps)].tail(1).to_numpy()[0, 1:]

        f_T = dataset.tail(1).to_numpy()[0, 1:]

        # minihack levels can be negative so let's scale up to 0
        f_T[f_T <= 0] = 0
        f_i[f_i <= 0] = 0
        f += (f_i - f_T)
        if verbose:
            print("Task {0}".format(i+1))
            print("Forgetting {0}, Return {1}, Final Return {2}".format((f_i - f_T), f_i, f_T))
    return f / num_tasks

def integrate(
    dataset: np.array,
    num_seeds: int,
    num_steps: int,
    task: int=None,
    aggregate: bool=False,
) -> np.array:
    # dataframe construction for different numbers of seeds
    df = {'t': dataset[:, 0]}
    for i in range(num_seeds):
        df['s{}'.format(i)] = dataset[:, i + 1]
    dataset = pd.DataFrame(df)

    # rectangle rule for integration
    upper = num_steps if task is None else task * num_steps
    lower = 0 if task is None else (task - 1) * num_steps
    dataset = dataset[(dataset['t'] >= lower) & (dataset['t'] < upper)].fillna(method='ffill').to_numpy()
    auc = np.zeros(1) if aggregate else np.zeros(num_seeds)
    for i in range(1, dataset.shape[0]):
        delta_t = dataset[i, 0] - dataset[i - 1, 0]
        f = dataset[i, 1:]
        # let's max 0 the smallest value of the performance
        f[f <= 0] = 0
        # take the mean of the performance across seeds
        if aggregate:
            f = np.mean(f)
        auc += delta_t * f
    auc /= num_steps
    return auc


def fwd_transfer(
    dones: Dict[str, np.array],
    dones_ref: Dict[str, np.array],
    num_tasks: int=8,
    num_seeds: int=10,
    envs: List = [
        "Room-Random-15x15-v0",
        "Room-Monster-15x15-v0",
        "Room-Trap-15x15-v0",
        "Room-Ultimate-15x15-v0",
        "River-Narrow-v0",
        "River-v0",
        "River-Monster-v0",
        "HideNSeek-v0"
    ],
    num_steps: int=1e6,
    full_range: bool=False,
    verbose: bool=False,
    aggregate_ref_auc: bool=False,
) -> np.array:
    """
    N = num_tasks
    ft = \sum_i=1^N ft_i
    ft_i = auc_i - auc / (1 - auc_i)
    auc_i = are under the curve for particular task in cl loop
    auc = area under the curve for the single task
    ft_i = auc_i - auc / (1 - auc_i)
    """

    ft = np.zeros(num_seeds)
    keys = list(dones.keys())
    for i, e in enumerate(envs):
        if e in list(dones_ref.keys()):
            ref_perf = dones_ref[e]
            key = keys[i]
            cl_perf = dones[key]
            ref_auc = integrate(ref_perf, num_seeds, num_steps, task=None, aggregate=aggregate_ref_auc)
            if full_range:
                cl_auc = integrate(cl_perf, num_seeds, num_steps * num_tasks, task=None, aggregate=False)
            else:
                cl_auc = integrate(cl_perf, num_seeds, num_steps, task=i + 1, aggregate=False)
            ft_i = (cl_auc - ref_auc) / (1 - cl_auc)
            ft += ft_i
            if verbose:
                print("{0} ref auc {1} auc {2}".format(e, ref_auc, cl_auc))
                print("per task ft {0}".format(ft_i))
    return ft / num_tasks
