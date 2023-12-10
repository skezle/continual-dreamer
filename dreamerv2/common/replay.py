import collections
import datetime
import io
import pathlib
import uuid
import os
import sys
import heapq
import random
import pickle

import numpy as np
import tensorflow as tf
import wandb

import time

def coverage_maximization_distance(tensor1: np.ndarray, tensor2: np.ndarray, distance_type: str):
    if distance_type == "euclid":
        return np.linalg.norm(tensor1 - tensor2)
    elif distance_type == "cosine":
        assert tensor1.ndim == 1 and  tensor2.ndim == 1, "wrong shape in Coverage Maximization distance"
        return (tensor1 @ tensor2.T)/(np.linalg.norm(tensor1)*np.linalg.norm(tensor2))
    else:
        print("Wrong distance type")

class Replay:
    def __init__(
        self,
        directory,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
        reservoir_sampling=False,
        reward_sampling=False,
        num_tasks=1,
        recent_past_sampl_thres=0,
        uncertainty_sampling=False,
        uncertainty_recalculation:int=5000,
        coverage_sampling=False,
        coverage_sampling_args=None
    ):
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity # replay buffer size
        self._ongoing = ongoing
        self._minlen = minlen  # this is used as a cutoff before storing in the replay buffer
        self._maxlen = maxlen  # only used to sample seq_lens
        self._prioritize_ends = prioritize_ends  # prioritizes the end of an epsiode which is loaded
        self._reservoir_sampling = reservoir_sampling  # whether to use reservoir sampling or not
        self.recent_past_sampl_thres = recent_past_sampl_thres  # probability above this threshold trigger uniform episode. Below trigger triangle distribution
        self._reward_sampling = reward_sampling  # whether to use rewrad sampling or not
        self._coverage_sampling = coverage_sampling # whether to use coverage maximization
        self._coverage_sampling_args = coverage_sampling_args # coverage maximization args
        self._num_tasks = num_tasks  # the number of tasks in the cl loop
        self._random = np.random.RandomState()
        # total_steps / eps is the total number of steps / eps seen over the course of training
        # loaded_steps / eps is the total number of steps / eps in the replay buffer
        # filename -> key -> value_sequence
        self._complete_eps, self._tasks, self._reward_eps = load_episodes(
            directory=self._directory,
            capacity=capacity,
            minlen=minlen,
            coverage_sampling=self._coverage_sampling,
            coverage_sampling_args=self._coverage_sampling_args,
            check=False)
        # worker -> key -> value_sequence
        self._ongoing_eps = collections.defaultdict(lambda: collections.defaultdict(list))
        self._total_episodes, self._total_steps = count_episodes(directory)
        self._loaded_episodes = len(self._complete_eps)
        self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())

        self._plan2explore = None
        self._uncertainty_sampling = uncertainty_sampling
        self._uncertainty_recalculation =  uncertainty_recalculation
        self._episodes_uncertainties = collections.defaultdict()
       
        self.set_task()
        if self._coverage_sampling:
            self.coverage_maximization_initialization(directory)
                
    def set_task(self, task_idx=0):
        self.task_idx = task_idx

    def get_task(self):
        return self.task_idx
    
    def get_max_task(self):
        if len(self._tasks) > 0:
            return np.max([v for k, v in self._tasks.items()])
        else:
            return 0
        
    def coverage_maximization_initialization(self, directory):
        self._replay_cell = tf.keras.layers.ConvLSTM2D(
            filters=self._coverage_sampling_args["filters"], kernel_size=self._coverage_sampling_args["kernel_size"],
            padding="same", return_sequences=False, return_state=True, activation="elu")
               
        self._episodes_heap = []
        self._lstm_states = {}
        if bool(self._complete_eps):
            for filename, episode in self._complete_eps.items():
                if self._coverage_sampling_args["normalize_lstm_state"]:
                    self._lstm_states[str(filename)] = self._replay_cell(tf.expand_dims(np.array(episode['image'])/255,axis=0))[2].numpy().reshape(-1)  # LSTM has return_state=True, so it returns three outputs, and last one is a cell state
                    self._lstm_states[str(filename)] /= np.linalg.norm(self._lstm_states[str(filename)])
                else:
                    self._lstm_states[str(filename)] = self._replay_cell(tf.expand_dims(np.array(episode['image'])/255,axis=0))[2].numpy().reshape(-1)  # LSTM has return_state=True, so it returns three outputs, and last one is a cell state
            
            with open(directory/f'coverage_max_heap.pkl', 'rb') as handle:
                self._episodes_heap = pickle.load(handle)

                
    @property
    def stats(self):
        ret = {
            'total_steps': self._total_steps,
            'total_episodes': self._total_episodes,
            'loaded_steps': self._loaded_steps,
            'loaded_episodes': self._loaded_episodes,
            'av_task': np.mean([v for k, v in self._tasks.items()]),
            'er_task': [len([v for k, v in self._tasks.items() if v == i]) for i in range(self._num_tasks)],
        }
        # max_task = self.get_max_task()
        # for i in range(max_task + 1):
        #     ret['er_task_{0}'.format(i)] = len([v for k, v in self._tasks.items() if v == i])
        return ret

    def add_step(self, transition, worker=0, task_index=0):
        episode = self._ongoing_eps[worker]
        for key, value in transition.items():
            episode[key].append(value)
        episode["task_index"] = task_index
        if transition['is_last']:
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):
        length = eplen(episode)
        if length < self._minlen:
            print(f'Skipping short episode of length {length}.')
            return
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: convert(value) for key, value in episode.items()}
        task = self.get_task()
        filename = save_episode(self._directory, episode, task, self._total_episodes)
        # add candidate to the replay buffer
        self._complete_eps[str(filename)] = episode
        self._tasks[str(filename)] = task
        self._reward_eps[str(filename)] = episode['reward'].astype(np.float64).sum()

        if self._total_episodes % self._uncertainty_recalculation == 0:
            self._episodes_uncertainties.clear()

        if self._coverage_sampling:
            if self._coverage_sampling_args["normalize_lstm_state"]:
                self._lstm_states[str(filename)] = self._replay_cell(tf.expand_dims(np.array(episode['image'])/255,axis=0))[2].numpy().reshape(-1)  # LSTM has return_state=True, so it returns three outputs, and last one is a cell state
                self._lstm_states[str(filename)] /= np.linalg.norm(self._lstm_states[str(filename)])
            else:
                self._lstm_states[str(filename)] = self._replay_cell(tf.expand_dims(np.array(episode['image'])/255,axis=0))[2].numpy().reshape(-1)  # LSTM has return_state=True, so it returns three outputs, and last one is a cell state

            if len(self._episodes_heap) == 0 and self._total_episodes != 0:            
                # In this line, we initialize the priority queue with some arbitrary priority value
                heapq.heappush(self._episodes_heap, (1, str(filename))) # A 'heapq' is a priority queue -- a special type of queue in which each element is associated with a priority value.
                self._complete_eps[str(filename)] = episode
            else:
                if self._loaded_steps < self._capacity:
                    start = time.time()
                    distances = [coverage_maximization_distance(self._lstm_states[str(filename)], self._lstm_states[str(replay_files)], self._coverage_sampling_args['distance'])
                                    for replay_files in np.random.choice(list(self._lstm_states.keys()), np.min((len(list(self._lstm_states.keys())),
                                        self._coverage_sampling_args["number_of_comparisons"])), replace=False)]
                    distance_metric = np.median(distances)
                    end = time.time() - start
                    heapq.heappush(self._episodes_heap, (distance_metric, str(filename)))
                    self._complete_eps[str(filename)] = episode
                elif self._loaded_steps >= self._capacity:
                    start = time.time()
                    distances = [coverage_maximization_distance(self._lstm_states[str(filename)], self._lstm_states[str(replay_files)], self._coverage_sampling_args['distance'])
                                for replay_files in np.random.choice(list(self._lstm_states.keys()),np.min((len(list(self._lstm_states.keys())),
                                        self._coverage_sampling_args["number_of_comparisons"])), replace=False)]
                    distance_metric = np.median(distances)
                    end = time.time() - start

                    # In the line below, we add a new episode with the corresponding distance metric to the heapq, and next, remove the episode with the smallest distance.
                    priority, filename_remove = heapq.heappushpop(self._episodes_heap,(distance_metric, str(filename)))
                    episode_remove = self._complete_eps[str(filename_remove)]
                    self._loaded_steps -= eplen(episode_remove)
                    self._loaded_episodes -= 1
                    del self._complete_eps[str(filename_remove)]
                    del self._tasks[str(filename_remove)]
                    del self._reward_eps[str(filename_remove)]
                    del self._lstm_states[str(filename_remove)]
                self._logger.scalar("replay_cm/total_steps", self._total_steps)
                self._logger.scalar("replay_cm/distances_time", end)
                self._logger.scalar("replay_cm/total_episodes", self._total_episodes)
                self._logger.scalar("replay_cm/distances_min", np.min(distances))
                self._logger.scalar("replay_cm/distances_max", np.max(distances))
                self._logger.scalar("replay_cm/distances_mean", np.mean(distances))
                self._logger.scalar("replay_cm/distances_median", np.median(distances))
                self._logger.scalar("replay_cm/distances_percentile75", np.percentile(distances, 75))
                self._logger.scalar("replay_cm/distances_percentile25", np.percentile(distances, 25))
                with open(self._directory/f'coverage_max_heap.pkl', 'wb') as handle:
                    pickle.dump(self._episodes_heap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif self._reservoir_sampling:
            # Alg 2 from https://arxiv.org/pdf/1902.10486.pdf
            if self._loaded_steps < self._capacity:
                self._complete_eps[str(filename)] = episode
            else:
                # self._total_episodes: the total number of episodes seen so far
                i = np.random.randint(self._total_episodes)
                # this condition is should be if i < mem_sz, mem_sz is in number of 
                # transitions, but experience is stored in terms of episodes
                # self._loaded_episodes is a surrogate

                # we need to correct self._loaded_episodes
                # since we have incremented self._loaded_episodes without adding a filename
                # to self._complete_eps
                if i < self._loaded_episodes:
                    # remove item i from the replay buffer
                    # it can be re-loaded if the run starts again
                    # so need store an additional dictionary to store on disk to keep track of
                    # of the reservoir.
                    filenames = [k for k, v in self._complete_eps.items()]
                    filename_remove = filenames[i]
                    episode_remove = self._complete_eps[str(filename_remove)]
                    self._loaded_steps -= eplen(episode_remove)
                    self._loaded_episodes -= 1
                    del self._complete_eps[str(filename_remove)]
                    del self._tasks[str(filename_remove)]
                    del self._reward_eps[str(filename_remove)]
                    if str(filename_remove) in self._episodes_uncertainties:
                        del self._episodes_uncertainties[str(filename_remove)]
            with open(self._directory/f'rs_buffer.pkl', 'wb') as handle:
                pickle.dump(list(self._complete_eps.keys()), handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._enforce_limit()

    def dataset(self, batch, length, oversampling=False):
        example = next(iter(self._generate_chunks(length, oversampling)))
        dataset = tf.data.Dataset.from_generator(
            lambda: self._generate_chunks(length, oversampling),
            {k: v.dtype for k, v in example.items()},
            {k: v.shape for k, v in example.items()})
        dataset = dataset.batch(batch, drop_remainder=True)
        dataset = dataset.prefetch(5)
        return dataset


    def _generate_chunks(self, length, oversampling):
        sequence = self._sample_sequence(oversampling)

        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding['action'])
                if len(sequence['action']) < 1:
                    sequence = self._sample_sequence(oversampling)
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            yield chunk

    def _sample_sequence(self, oversampling):
        episodes_keys = list(self._complete_eps.keys())
        episodes = list(self._complete_eps.values())
        if self._ongoing:
            episodes_keys += [k for k, v in self._ongoing_eps.items() if eplen(v) >= self._minlen]
        if self._ongoing:
            episodes += [
                x for x in self._ongoing_eps.values()
                if eplen(x) >= self._minlen]
        if self._reward_sampling:
            rewards = list(self._reward_eps.values())
            # if there is a mismatch in lengths lets sync the rewards with self._complete_eps()
            if len(rewards) != len(episodes_keys):
                print("Syncing eps _reward_eps and _complete_eps")
                _, _, self._reward_eps = load_episodes(self._directory,
                                                       self.capacity, self.minlen, self._coverage_sampling,
                                                       self._coverage_sampling_args, check=False)
                rewards = list(self._reward_eps.values())
            e_r = np.exp(rewards - np.max(rewards))
            rewards_norm = e_r / e_r.sum()
            episode_key = self._random.choice(episodes_keys, p=rewards_norm)
        elif self.recent_past_sampl_thres > np.random.random():
            episode_key = episodes_keys[
                int(np.floor(np.random.triangular(0, len(episodes_keys), len(episodes_keys), 1)))
            ]
        elif self._uncertainty_sampling:
            self._check_if_uncertainties_available()
            uncertainties = np.array(list(self._episodes_uncertainties.values()))
            e_unc = np.exp(uncertainties - np.max(uncertainties))
            uncertainty_norm = e_unc / e_unc.sum()
            episode_key = np.random.choice(
                list(self._episodes_uncertainties.keys()),
                p=uncertainty_norm
            )
            self._logger.scalar("replay/uncertainty", self._episodes_uncertainties[episode_key])
        elif oversampling:
            episodes_0 = []
            episodes_1 = []

            for epi in episodes:
                if epi["task_index"] == 0:
                    episodes_0.append(epi)
                elif epi["task_index"] == 1:
                    episodes_1.append(epi)

            if np.random.uniform(0, 1) < 0.99 and len(episodes_1) > 0:
                i = np.random.randint(0, len(episodes_1))
                episode = episodes_1[i]
            elif len(episodes_0) > 0:
                i = np.random.randint(0, len(episodes_0))
                episode = episodes_0[i]
            else:
                i = np.random.randint(0, len(episodes_1))
                episode = episodes_1[i]

        else:
            episode_key = self._random.choice(episodes_keys)
            episode = self._complete_eps[episode_key]
        total = len(episode['action'])
        length = total
        if self._maxlen:
            length = min(length, self._maxlen)
        # Randomize length to avoid all chunks ending at the same time in case the
        # episodes are all of the same length.
        length -= np.random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1
        if self._prioritize_ends:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)

        sequence = {
            k: convert(v[index: index + length])
            for k, v in episode.items() if not k.startswith(('log_', "task_index"))}
        sequence['is_first'] = np.zeros(len(sequence['action']), np.bool)
        sequence['is_first'][0] = True
        if self._maxlen:
            assert self._minlen <= len(sequence['action']) <= self._maxlen

        return sequence

    def _check_if_uncertainties_available(self):
        keys_to_use = list(self._complete_eps.keys())

        # Check if we have uncertanity value for every episode
        for i, ep_key in enumerate(keys_to_use):
            if self._episodes_uncertainties.get(ep_key, None) is None:
                ep_expanded = {
                    key: np.expand_dims(elem, 0)
                    for key, elem in self._complete_eps[ep_key].items()
                }
                inputs = self.agent.wm.forward(ep_expanded, None)

                if self.agent._task_behavior.config.disag_action_cond:
                    action = tf.cast(ep_expanded["action"], inputs.dtype)
                    inputs = tf.stop_gradient(tf.concat([inputs, action], -1))

                preds = [head(inputs).mode() for head in self.agent._expl_behavior._networks]
                disag = tf.Tensor(preds).std(0).mean(-1)
                ep_uncertainty = np.mean(disag.cpu().numpy())
                self._episodes_uncertainties[ep_key] = ep_uncertainty

    def _enforce_limit(self):
        if not self._capacity:
            return
        while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
            # Relying on Python preserving the insertion order of dicts.
            if self._coverage_sampling:
                _, candidate = heapq.heappop(self._episodes_heap)
                episode = self._complete_eps[str(candidate)]
                del self._lstm_states[str(candidate)]
            elif self._reservoir_sampling:
                candidate, episode = random.sample(self._complete_eps.items(), 1)[0]
            else:
                # Relying on Python preserving the insertion order of dicts.
                # first-in-first-out
                candidate, episode = next(iter(self._complete_eps.items()))
            self._loaded_steps -= eplen(episode)
            self._loaded_episodes -= 1
            del self._complete_eps[str(candidate)]
            del self._tasks[str(candidate)]
            del self._reward_eps[str(candidate)]
            if str(candidate) in self._episodes_uncertainties:
                del self._episodes_uncertainties[str(candidate)]
        
        if self._coverage_sampling:
            with open(self._directory/f'coverage_max_heap.pkl', 'wb') as handle:
                pickle.dump(self._episodes_heap, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self._reservoir_sampling:
            with open(self._directory/f'rs_buffer.pkl', 'wb') as handle:
                pickle.dump(list(self._complete_eps.keys()), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def agent(self):
        return self._plan2explore

    @agent.setter
    def agent(self, plan2explore):
        self._plan2explore = plan2explore

    @property
    def logger(self):
        return self._logger

    @agent.setter
    def logger(self, logger):
        self._logger = logger 
    
def count_episodes(directory):
    filenames = list(directory.glob("*.npz"))
    num_episodes = len(filenames)

    if num_episodes > 0:
        assert (
            len(str(os.path.basename(filenames[0])).split("-")) == 5
        ), "Probably filenames are not in following format: f'{timestamp}-{identifier}-{task}-{length}-{total_episodes}.npz'"
    num_steps = sum(int(str(os.path.basename(n)).split("-")[3]) - 1 for n in filenames)
    return num_episodes, num_steps



def save_episode(directory, episode, task, total_episodes):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f'{timestamp}-{identifier}-{task}-{length}-{total_episodes}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename


def load_episodes(directory, capacity=None, minlen=1, coverage_sampling=False, coverage_sampling_args=None, check=False):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob('*.npz'))
    if coverage_sampling and os.path.exists(directory/f'coverage_max_heap.pkl'):
        with open(directory/f'coverage_max_heap.pkl', 'rb') as handle:
            _episodes_heap = pickle.load(handle)
        filenames = list(zip(*_episodes_heap))[1]

    if capacity:
        num_steps = 0
        num_episodes = 0
        # we are going to only fetch the most recent
        # if we are doing reservoir sampling a random shuffle of the replay buffer 
        # will preserve and uniform distribution over all tasks
        if os.path.exists(directory/f'rs_buffer.pkl'):
            with open(directory/f'rs_buffer.pkl', 'rb') as handle:
                filenames = pickle.load(handle)
            filenames = [pathlib.Path(filename) for filename in filenames]
        for filename in reversed(filenames):
            length = int(str(os.path.basename(filename)).split('-')[3])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    tasks = {}
    rewards_eps = {}
    for filename in filenames:
        try:
            with filename.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        episodes[str(filename)] = episode
        task = int(str(os.path.basename(filename)).split('-')[2])
        tasks[str(filename)] = task
        rewards_eps[str(filename)] = episode['reward'].astype(np.float64).sum()

    # Collas rebuttal check for duplicate episodes
    # First run a CL run without deleteing the replay buffer
    # Then re-run it again with the the check flag manually turned on
    if check:
        i = 0
        for filename, episode in episodes.items():
            print("[{0} / {1}] eps checked".format(i, len(episodes)))
            episodes_comparison = episodes.copy()
            j = 0
            for filename2, episode2 in episodes_comparison.items():
                # only select episodes which we haven't compared to
                # also let's not compare the episode to itself
                if j <= i:
                    j += 1
                    continue

                episode_elements_equals = 0
                for key, value in episode2.items():
                    if np.array_equal(episode[key], value):
                        episode_elements_equals += 1
                    else:
                        break
                if episode_elements_equals == len(episode2):
                    raise ValueError(f'Episode {filename} and {filename2} are the same')
                j += 1
            i += 1
        sys.exit("Finished check")
    
    return episodes, tasks, rewards_eps

def parse_episode_name(episode_name):
    episode_name = os.path.basename(episode_name)
    parts = episode_name.split("-")
    timestamp = parts[0]
    identifier = parts[1]
    task = parts[2]
    length = parts[3]
    total_episodes = parts[4] if len(parts) == 5 else None
    if len(parts) == 5:
        total_episodes = total_episodes.split(".")[0]
    elif len(parts) == 4:
        length = length.split(".")[0]

    return {
        "timestamp": timestamp,
        "identifier": identifier,
        "task": int(task),
        "length": int(length),
        "total_episodes": int(total_episodes) if total_episodes else np.nan,
    }

def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


def eplen(episode):
    return len(episode['action']) - 1
