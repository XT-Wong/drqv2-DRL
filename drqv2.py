# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import utils


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
                 use_r3m, use_vip, use_optical_flow):
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
        

        # models
        if use_r3m:
            from r3m import load_r3m
            self.encoder = load_r3m("resnet50").to(device)
            self.encoder.eval()
            self.repr_dim = 2048 * 3
        elif use_vip:
            from vip import load_vip
            self.encoder = load_vip().to(device)
            self.encoder.eval()
            self.repr_dim = 1024 * 3
        else:
            self.encoder = Encoder(obs_shape).to(device)
            self.repr_dim = self.encoder.repr_dim
        self.actor = Actor(self.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

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
        self.encoder.train(training)
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

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
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

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
