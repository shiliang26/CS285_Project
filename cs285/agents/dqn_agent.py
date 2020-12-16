import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, env_2, agent_params):

        self.env = env
        self.env_2 = env_2
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()
        self.last_obs_2 = self.env_2.reset()
        self.pretrained = agent_params['pretrained']

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        if self.agent_params['pretrained']:
            self.learning_starts = 0
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.replay_buffer_idx_2 = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'])
        self.replay_buffer_2 = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'])
        self.t = 0
        if self.agent_params['pretrained']:
            self.t = self.agent_params['starting_point']
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass


    def switch_environment(self, env_num):
        self.critic.q_net.head = env_num
        self.critic.q_net_target.head = env_num


    def step_env(self):

        # Pacman
        self.switch_environment(1)
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        eps = self.exploration.value(self.t)
        perform_random_action = np.random.random() < eps or self.t < self.learning_starts

        if perform_random_action:
            action = np.random.randint(9)
        else:
            obs = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(obs).squeeze()
        last_obs, reward, done, info = self.env.step(action)
        self.last_obs = last_obs
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        if done:
            self.last_obs = self.env.reset()

        # Alien
        self.switch_environment(2)
        self.replay_buffer_idx_2 = self.replay_buffer_2.store_frame(self.last_obs_2)
        eps = self.exploration.value(self.t)
        perform_random_action = np.random.random() < eps or self.t < self.learning_starts

        if perform_random_action:
            action = np.random.randint(18)
        else:
            obs = self.replay_buffer_2.encode_recent_observation()
            action = self.actor.get_action(obs).squeeze()
        last_obs, reward, done, info = self.env_2.step(action)
        self.last_obs_2 = last_obs
        self.replay_buffer_2.store_effect(self.replay_buffer_idx_2, action, reward, done)
        if done:
            self.last_obs_2 = self.env_2.reset()


    def sample(self, batch_size, env):
        if env == 1 and self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        elif env == 2 and self.replay_buffer_2.can_sample(self.batch_size):
            return self.replay_buffer_2.sample(batch_size)
        else:
            return [],[],[],[],[]


    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
                and self.replay_buffer_2.can_sample(self.batch_size)
        ):

            log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        return log

    def train_distill(self, ob_no, env_num):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
                and self.replay_buffer_2.can_sample(self.batch_size)
        ):

            log = self.critic.distill(ob_no, env_num)

        return log