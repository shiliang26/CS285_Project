import torch
import numpy as np
from torchvision.utils import save_image
from cs285.infrastructure.gradcam import GradCAM, visualize_cam


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic
        self.cam = GradCAM(self.critic.q_net)
        self.savepath = '/content/cs285_project/data/cams/multi/'
        self.num_save = 0

    def get_action(self, obs, cam_flag):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        observation = ptu.from_numpy(observation)
        # q_values = self.critic.q_net(observation)       # q_values: (batch, num_actions)
        if cam_flag:
            mask, q_values = self.cam(observation)
        else:
            q_values = self.critic.q_net(observation)

        q_values = ptu.to_numpy(q_values)

        action = np.argmax(q_values, axis = 1)

        # observation: [1, 84, 84, 4]
        if cam_flag:
            channel = observation.permute(3, 0, 1, 2)[0]
            torch_img = channel.float().div(255)
            heatmap, result = visualize_cam(mask, torch_img)
            save_image(result, self.savepath + str(self.num_save) + '.png')
            self.num_save += 1

        return action