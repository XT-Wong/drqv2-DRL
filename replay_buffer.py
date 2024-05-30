# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


# class PrioritizedReplayBuffer(ReplayBuffer):
#     def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
#                  fetch_every, save_snapshot, eps=0.01, alpha=0.7, beta=0.4):
#         print('init PER')
#         # self.priorities = np.zeros(max_size, dtype=np.float32)
#         self.eps = eps  # minimal priority for stability
#         self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
#         self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
#         self.max_priority = eps  # priority for new samples, init as eps
#         self.priorities = dict()
#         self.str2fn = dict()
#         self.file_priorities = dict()
#         self.max_file_priorities = eps
#         self.file_max_priorities = dict()
#         super().__init__(replay_dir, max_size, num_workers, nstep, discount,
#                          fetch_every, save_snapshot)
    
#     # def __copy__(self):
#     #     print('!!!!!')
#     #     return None

#     def _sample_episode(self):
#         # eps_fn = random.choice(self._episode_fns)
#         # return self._episodes[eps_fn]
#         eps_fn = random.choices(list(self.file_priorities.keys()), weights=list(self.file_priorities.values()))
#         # print(self.file_priorities)
#         return self._episodes[eps_fn[0]], eps_fn[0]

#     def _store_episode(self, eps_fn):
#         try:
#             episode = load_episode(eps_fn)
#         except:
#             return False
#         eps_len = episode_len(episode)
#         while eps_len + self._size > self._max_size:
#             early_eps_fn = self._episode_fns.pop(0)
#             early_eps = self._episodes.pop(early_eps_fn)
#             # PER pop
#             self.priorities.pop(early_eps_fn)
#             self.file_priorities.pop(early_eps_fn)
#             self.file_max_priorities.pop(early_eps_fn)
#             self.str2fn.pop(str(early_eps_fn))
#             self._size -= episode_len(early_eps)
#             early_eps_fn.unlink(missing_ok=True)
#         self._episode_fns.append(eps_fn)
#         self._episode_fns.sort()
#         self._episodes[eps_fn] = episode
#         self._size += eps_len

#         # PER ADD
#         self.priorities[eps_fn] = np.ones(eps_len, dtype=np.float32) * self.max_priority
#         self.file_priorities[eps_fn] = self.max_priority * eps_len
#         self.file_max_priorities[eps_fn] = self.max_priority
#         self.str2fn[str(eps_fn)] = eps_fn
#         # print(self.str2fn, 'after fetch')
#         self.max_file_priorities = max(list(self.file_max_priorities.values()))

#         if not self._save_snapshot:
#             eps_fn.unlink(missing_ok=True)
#         return True

#     def _try_fetch(self):
#         if self._samples_since_last_fetch < self._fetch_every:
#             return
#         self._samples_since_last_fetch = 0
#         try:
#             worker_id = torch.utils.data.get_worker_info().id
#         except:
#             worker_id = 0
#         eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
#         fetched_size = 0
#         for eps_fn in eps_fns:
#             eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
#             if eps_idx % self._num_workers != worker_id:
#                 continue
#             if eps_fn in self._episodes.keys():
#                 break
#             if fetched_size + eps_len > self._max_size:
#                 break
#             fetched_size += eps_len
#             if not self._store_episode(eps_fn):
#                 break

#     def _sample(self):
#         try:
#             self._try_fetch()
#         except:
#             traceback.print_exc()
#         self._samples_since_last_fetch += 1
#         episode, eps_fn = self._sample_episode()
#         # add +1 for the first dummy transition
#         # idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
#         idx = np.random.choice(range(1, episode_len(episode) - self._nstep + 2), p=self.priorities[eps_fn][1: episode_len(episode) - self._nstep + 2] / self.priorities[eps_fn][1: episode_len(episode) - self._nstep + 2].sum())
#         obs = episode['observation'][idx - 1]
#         action = episode['action'][idx]
#         next_obs = episode['observation'][idx + self._nstep - 1]
#         reward = np.zeros_like(episode['reward'][idx])
#         discount = np.ones_like(episode['discount'][idx])

#         # PER get weight
#         p = self.priorities[eps_fn] / sum(self.file_priorities.values())
#         p_i = p[idx]
#         w_max = -1
#         for fn in list(self.priorities.keys()):
#             w_tmp = np.max((1 / self._size / self.priorities[fn]) ** self.beta, axis=-1)
#             w_max = np.max([w_max, w_tmp], axis=-1)
#         weight = (1 / self._size / p_i) ** self.beta / w_max

#         for i in range(self._nstep):
#             step_reward = episode['reward'][idx + i]
#             reward += discount * step_reward
#             discount *= episode['discount'][idx + i] * self._discount
        
#         try:
#             worker_id = torch.utils.data.get_worker_info().id
#         except:
#             worker_id = 0
#         # print(self.str2fn, 'after sample')
#         return (obs, action, reward, discount, next_obs), weight, worker_id, str(eps_fn), idx

#     def update_priorities(self, eps_fn_list, idxs_list, priorities: np.ndarray):
#         update_file = dict()
#         priorities = (priorities + self.eps) ** self.alpha
#         try:
#             worker_id = torch.utils.data.get_worker_info().id
#         except:
#             worker_id = 0
        
#         # print(self.str2fn, self.file_priorities, self._episodes, worker_id)
#         for i in range(len(eps_fn_list[0])):
#             file_worker = eps_fn_list[0][i]
#             if file_worker != worker_id:
#                 continue
#             fake_eps_fn = eps_fn_list[1][i]
#             eps_fn = self.str2fn[fake_eps_fn]
#             idx = idxs_list[i]
#             self.priorities[eps_fn][idxs] = priorities[i]
#             update_file[eps_fn] = 1
#         for eps_fn in list(update_file.keys()):
#             self.file_max_priorities[eps_fn] = max(self.priorities[eps_fn])
#             self.file_priorities[eps_fn] = sum(self.priorities[eps_fn])
#         self.max_priorities = max(list(self.file_max_priorities.values()))

#     def __iter__(self):
#         while True:
#             # print(self.str2fn, 'itering')
#             yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)
    # iterable = PrioritizedReplayBuffer(replay_dir,
    #                                     max_size_per_worker,
    #                                     num_workers,
    #                                     nstep,
    #                                     discount,
    #                                     fetch_every=1000,
    #                                     save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
