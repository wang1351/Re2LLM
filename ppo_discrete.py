import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import pdb
import random

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.outdim = args.action_dim
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width).cuda()
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width).cuda()
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim).cuda()
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh].cuda()  

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = s.cuda()
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width).cuda()
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width).cuda()
        self.fc3 = nn.Linear(args.hidden_width, 1).cuda()
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh].cuda()

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = s.cuda()
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_discrete:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.savedir = args.savedir
        self.actor = Actor(args)

        self.critic = Critic(args)
        if self.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).cuda()
        a_prob = self.actor(s).detach().cpu().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).cuda()
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
            a = a.item()
            if random.random() > 0.9:
                a = random.randint(0, self.actor.outdim-1)
            return a, a_logprob.item()

    def eval_choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).cuda()
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
            a = a.item()
            return a, a_logprob.item()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()
        adv = []
        gae = 0
        with torch.no_grad():
            vs = self.critic(s).cpu()
            vs_ = self.critic(s_).cpu()

            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs

        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze().cuda()).view(-1, 1)
                ratios = torch.exp(a_logprob_now - a_logprob[index].cuda())

                surr1 = ratios * adv[index].cuda()
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index].cuda()
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index]).cuda()
                critic_loss = F.mse_loss(v_target[index].cuda(), v_s)
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save(self, total):
        torch.save(self.actor.state_dict(), './'+ self.savedir +'_epoch_'+ str(total) + '.pth')
