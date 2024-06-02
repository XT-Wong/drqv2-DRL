# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import cv2

import torchvision.transforms as T
import hashlib
import utils

class CLOPLayer(nn.Module):
    def __init__(self, p=0.9):
        super().__init__()
        self.p = p

    def _shuffle(self, x):
        batch_size = x.shape[0]
        nb_channels = x.shape[1]
        flat = x.flatten(start_dim=2, end_dim=-1)
        idx = self._index_permute(x[0, 0]).to(x.device)
        idx = idx.repeat(batch_size, nb_channels, 1)
        res = torch.gather(flat, 2, idx)
        return res.view_as(x)

    def _index_permute(self, x):
        n_element = x.nelement()
        dim = int(np.sqrt(n_element))
        permuted_indexes = torch.arange(0, n_element, dtype=int)
        p = (1 - self.p / 2, self.p / 8, self.p / 8, self.p / 8, self.p / 8)
        for i in range(-n_element + 1, n_element):
            i = abs(i)
            r = np.random.choice([0, 1, 2, 3, 4], p=p)
            if r != 0:
                if r == 1:
                    idx = i + 1
                if r == 2:
                    idx = i - 1
                if r == 3:
                    idx = i + dim
                if r == 4:
                    idx = i - dim
                if (idx > 0) & (idx < n_element):
                    tmp = int(permuted_indexes[i])
                    permuted_indexes[i] = int(permuted_indexes[idx])
                    permuted_indexes[idx] = tmp
        return permuted_indexes

    def forward(self, x):
        # print("shuffled")
        shuffled = self._shuffle(x)
        return shuffled
    

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)



class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):

        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)

        h = h.view(h.shape[0], -1)
        
        return h

class PadResizePlus(nn.Module):
    def __init__(self, highest_pad_strength):
        super().__init__()
        self.highest_pad_strength = int(highest_pad_strength)

    def crop(self, imgs, pad_x, pad_y):
        n, c, h_pad, w_pad = imgs.size()

        # calculate the crop size
        crop_x = w_pad - pad_x
        crop_y = h_pad - pad_y

        # create a grid for cropping
        eps_x = 1.0 / w_pad
        eps_y = 1.0 / h_pad
        x_range = torch.linspace(-1.0 + eps_x, 1.0 - eps_x, w_pad, device=imgs.device, dtype=imgs.dtype)[:crop_x]
        y_range = torch.linspace(-1.0 + eps_y, 1.0 - eps_y, h_pad, device=imgs.device, dtype=imgs.dtype)[:crop_y]

        grid_y, grid_x = torch.meshgrid(y_range, x_range)

        base_grid = torch.stack([grid_x, grid_y], dim=-1)
        # print('base_grid.shape', base_grid.shape)

        shift_x = torch.randint(0, pad_x + 1, size=(n, 1, 1, 1), device=imgs.device, dtype=imgs.dtype)
        shift_y = torch.randint(0, pad_y + 1, size=(n, 1, 1, 1), device=imgs.device, dtype=imgs.dtype)
        shift_x *= 2.0 / w_pad
        shift_y *= 2.0 / h_pad
        shift = torch.cat([shift_x, shift_y], dim=-1)
        grid = base_grid + shift
        
        # apply the grid to the input tensor to perform cropping
        padded_imgs_after_crop = F.grid_sample(imgs, grid)

        return padded_imgs_after_crop

    def forward(self, imgs):
        strength = torch.randint(0, self.highest_pad_strength+1, (1,)).item()
        
        _, _, h, w = imgs.shape
        pad_x = torch.randint(0, strength+1, (1,)).item()
        pad_y = strength - pad_x
        # [x+2*pad_x, y+2*pad_y]
        padded_imgs_before_crop = F.pad(imgs, (pad_x, pad_x, pad_y, pad_y))
        # print('padded_imgs_before_crop', padded_imgs_before_crop.shape)
        # [x+pad_x, y+pad_y]
        padded_imgs_after_crop = self.crop(padded_imgs_before_crop, pad_x, pad_y)
        # print('padded_imgs_after_crop', padded_imgs_after_crop.shape)
        # print('######################')

        resize = T.Resize(size=(h, w))

        return resize(padded_imgs_after_crop)


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):

        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2



class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 use_r3m, use_vip, use_optical_flow, encoder_eval, CLOP, use_CycAug,
                 actor_update_delay, with_target=False, actor_target_tau=0):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
        self.use_r3m = use_r3m
        self.use_vip = use_vip
        self.use_optical_flow = use_optical_flow
        self.encoder_eval = encoder_eval
        self.clop = CLOP
        self.cloplayer = CLOPLayer(self.clop)
        self.use_CycAug = use_CycAug
        self.aug_padcrop = RandomShiftsAug(pad=4)
        self.aug_padresize = PadResizePlus(highest_pad_strength=16)
        
        self.with_target = with_target
        self.actor_update_delay = actor_update_delay

        # models
        print("CLOP prob:", self.clop)
        print("use CycAug:", self.use_CycAug)
        if use_r3m:
            from r3m import load_r3m
            self.encoder = load_r3m("resnet50").to(device)
            self.repr_dim = 2048 * 3
        elif use_vip:
            from vip import load_vip
            
            self.encoder = load_vip().to(device)
            print("VIP loaded")
            self.encoder.eval()
            self.repr_dim = 1024 * 3
        else:
            self.encoder = Encoder(obs_shape).to(device)
            self.repr_dim = self.encoder.repr_dim

        self.actor = Actor(self.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        if self.with_target:
            self.actor_target_tau = actor_target_tau
            self.actor_target = deepcopy(self.actor).to(device)

        self.critic = Critic(self.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.eval() if self.encoder_eval else self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def obs_to_optical_flow(self, obs, obs_type="numpy", batched=False):
        def obs_to_optical_flow_once(obs:np.ndarray):
            obs = obs.reshape(-1, 3, 84, 84)
            grey_obs = []
            for i in range(obs.shape[0]):
                grey_obs.append(cv2.cvtColor(obs[i].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY))
            flows = [cv2.calcOpticalFlowFarneback(grey_obs[i], grey_obs[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(grey_obs)-1)]
            rst = np.zeros((84, 84, (obs.shape[0] - 1) * 3), dtype=np.uint8) # one frame all zero to keep shape
            for i in range(len(flows)):
                flow = flows[i]
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                # remove too small magnitude and angle
                magnitude[magnitude < 0.4] = 0
                angle[magnitude < 0.4] = 0
                rst[..., 3 * i] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                rst[..., 3 * i + 1] = angle * 180 / np.pi / 2
            rst = rst.transpose(2, 0, 1)
            # for i in range(obs.shape[0]):
            #     cv2.imwrite(f"/root/autodl-tmp/code/drqv2-DRL/obs_{i}.png", obs[i].transpose(1, 2, 0))
            # cv2.imwrite("/root/autodl-tmp/code/drqv2-DRL/flow_0.png", rst[:3].transpose(1, 2, 0))
            # cv2.imwrite("/root/autodl-tmp/code/drqv2-DRL/flow_1.png", rst[3:6].transpose(1, 2, 0))
            obs = np.concatenate([rst,obs[-1]], axis=0)
            return obs
        
        if obs_type == "torch":
            obs = obs.cpu().numpy()
        if batched:
            obs = np.array([obs_to_optical_flow_once(obs[i]) for i in range(obs.shape[0])]) # (batch, 9, 84, 84)
        else:
            obs = obs_to_optical_flow_once(obs)
        if obs_type == "torch":
            obs = torch.tensor(obs, device=self.device)
        return obs

    def act(self, obs, step, eval_mode):
        # if self.use_optical_flow:
        #     obs = self.obs_to_optical_flow(obs)
        obs = torch.as_tensor(obs, device=self.device)
        if self.use_r3m:
            # split into 3 images
            obs = obs.view(-1, 3, 84, 84)
            obs = self.encoder(obs)
            # concat
            obs = obs.view(-1, 3 * 2048)
        elif self.use_vip:
            obs = obs.view(-1, 3, 84, 84)
            obs = self.encoder(obs)
            obs = obs.view(-1, 3 * 1024)
        else:
            obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            if self.with_target:
                dist = self.actor_target(next_obs, stddev)
            else:
                dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        # print('0', replay_loader.dataset.str2fn, step)
        batch = next(replay_iter)
        # print('1', replay_loader.dataset.str2fn, step)

        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        # augment
        
        
        
        if self.use_CycAug:
            if (step // 200000) % 2 == 0:
                augmentation = self.aug_padcrop
            else:
                augmentation = self.aug_padresize

            obs = augmentation(obs.float())
            next_obs = augmentation(next_obs.float())
        else:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            
        # encode
        if self.use_r3m or self.use_vip:
            obs = obs.view(-1, 3, 84, 84)
            next_obs = next_obs.view(-1, 3, 84, 84)
        obs = self.encoder(obs)
        
        
        
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
        
        if self.use_r3m:
            obs = obs.view(-1, 3 * 2048)
            next_obs = next_obs.view(-1, 3 * 2048)
        elif self.use_vip:
            obs = obs.view(-1, 3 * 1024)
            next_obs = next_obs.view(-1, 3 * 1024)
            
        elif self.clop > 0:
            obs = obs.view(-1, 32, 35, 35)
            obs = self.cloplayer(obs)
            obs = obs.view(obs.shape[0], -1)
            
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        met = self.update_critic(obs, action, reward, discount, next_obs, step)
        metrics.update(met)

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)


        # update actor
        if step % (self.update_every_steps * self.actor_update_delay) == 0:
            metrics.update(self.update_actor(obs.detach(), step))

            # update actor target
            if self.with_target:
                utils.soft_update_params(self.actor, self.actor_target,
                                        self.actor_target_tau)
            metrics.update(self.update_actor(obs.detach(), step))
        return metrics
