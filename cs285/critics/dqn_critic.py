from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import torch.nn.functional as F

from cs285.infrastructure import pytorch_util as ptu


class DQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']
        self.tau = 0.01            # This is for distillation

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.pretrained = hparams['pretrained']

        if self.pretrained:
            self.q_net.load('multi.tar')
            self.q_net_target.load('multi.tar')

        if self.distill:
            self.teacher_1 = network_initializer(self.ob_dim, self.ac_dim)
            self.teacher_2 = network_initializer(self.ob_dim, self.ac_dim)
            self.teacher_1.load('teacher_pacman.tar')
            self.teacher_2.load('teacher_alien.tar')

        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):

        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        qa_t_values = self.q_net(ob_no)                                             # This generates Q(s, a) for all a
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)    # Picks out the Q(s, a) for the a we want
        
        qa_tp1_values = self.q_net_target(next_ob_no)

        if self.double_q:
            q_t_values_next = self.q_net(next_ob_no)
            _, acts = q_t_values_next.max(dim=1)                                       # Get Best Action from this network
            q_tp1 = torch.gather(qa_tp1_values, 1, acts.unsqueeze(1)).squeeze(1)     # Get Value from target network

        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)             # Gets the value to calculate targets

        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)


    def distill(self, ob_no, env_num):

        ob_no = ptu.from_numpy(ob_no)

        student = self.q_net(ob_no)
        if env_num == 1:
            teacher = self.teacher_1(ob_no)
        else:
            teacher = self.teacher_2(ob_no)

        loss = torch.sum(F.softmax(teacher / self.tau) * torch.log(F.softmax(teacher / self.tau) / F.softmax(student)))

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        return {
            'Distillation KL Loss': ptu.to_numpy(loss),
        }