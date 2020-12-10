import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        q_values = self.critic.qa_values(observation)       # q_values: (batch, num_actions)
        action = np.argmax(q_values, axis = 1)

        return action