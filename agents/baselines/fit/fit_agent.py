import copy
import logging
import os
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary

from helpers import utils
from helpers.utils import stack_on_channel, sample_frames
from rlbench import ObservationConfig
from rlbench.utils import get_stored_demos
from PIL import Image

from helpers.clip.core.clip import build_model, load_clip

NAME = 'FITAgent'
REPLAY_ALPHA = 0.7
REPLAY_BETA = 1.0

def get_attr(obs, obs_attr):
    return obs.obs_attr

class Actor(nn.Module):

    def __init__(self, actor_network: nn.Module):
        super(Actor, self).__init__()
        self._actor_network = copy.deepcopy(actor_network)
        self._actor_network.build()

    def train_module(self):
        self._actor_network.train_module()

    def get_optim_param_group(self, lr):
        return self._actor_network.get_optim_param_group(lr)

    def forward(self, observations, robot_state, lang_goal_desc=None, video_specification=None, image_goal_specification=None):
        mu = self._actor_network(observations, robot_state, lang_goal_desc=lang_goal_desc, video=video_specification, goal_image=image_goal_specification)
        return mu


class FITAgent(Agent):

    def __init__(self,
                 actor_network: nn.Module,
                 camera_name: str,
                 lr: float = 0.01,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 20.0,
                 task_specification_path: str = None,
                 use_lang_goal: bool = False,
                 use_video_goal: bool = False,
                 use_image_goal: bool = False,
                 observation_config: ObservationConfig = None):
        self._camera_name = camera_name
        self._actor_network = actor_network
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        assert os.path.isdir(task_specification_path), "Task specification directory {} does not exists".format(task_specification_path)
        self._task_specification_path = task_specification_path
        self._observation_config = observation_config
        self._use_lang_goal = use_lang_goal
        self._use_video_goal = use_video_goal
        self._use_image_goal = use_image_goal

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self._actor = Actor(self._actor_network).to(device).train(training)
        if training:
            self._sampling_strategy = 'rand'
            self._actor.train_module() ## Necessary to add since lang_model is initialized with .eval()
            params = self._actor.get_optim_param_group(self._lr)
            self._actor_optimizer = torch.optim.Adam(
                #self._actor.parameters(), lr=self._lr,
                params,
                weight_decay=self._weight_decay)
            logging.info('# Actor Params: %d' % sum(
                p.numel() for p in self._actor.parameters() if p.requires_grad))
        else:
            # No need for separate eval_module()
            self._sampling_strategy = 'uniform'
            for p in self._actor.parameters():
                p.requires_grad = False
            self._actor.eval()

        self._device = device

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        for p in self._actor.parameters():
            if (not(p.grad is None)) and torch.any(torch.isnan(p.grad)):
                print("grad values:", p.grad)
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    def batch_lang_goal_desc(self, lang_goals):
        if not type(lang_goals) is list:
            lang_goals = lang_goals.tolist()
        final_goals = [goal[0] if (len(goal)==1) else ' '.join(goal) for goal in lang_goals]
        return final_goals


    def get_visual_specification(self, tasks, task_variations): ## TODO: Optimize this function
        #print("task name:", tasks, "task_variation:", task_variations)
        cam_names = [self._camera_name] ## Future compatibility?
        num_cams = len(cam_names);
        t = 4 ## Hard coded to video of length 4
        video = np.zeros((tasks.shape[0], num_cams, t, 3, 256, 256)) ## current videos will be stored as [b, num_cams, t, 3, h, w]. However, model expects an input of [bs, t, 3, h, w]
        image_goal = np.zeros((tasks.shape[0], num_cams, 1, 3, 256, 256)) ## current videos will be stored as [b, num_cams, t, 3, h, w]. However, model expects an input of [bs, t, 3, h, w]
        for ind, (task_name, task_variation) in enumerate(zip(tasks, task_variations)):
            #print(task_name, task_variation)
            demo = get_stored_demos(
                                amount=1,
                                image_paths=True,
                                dataset_root=str(self._task_specification_path),
                                variation_number=int(task_variation[0]),
                                task_name=task_name[0],
                                random_selection=False,
                                obs_config=self._observation_config,
                            )[0]

            video_spec_task_i = []
            image_spec_task_i = []
            for cam_name in cam_names:
                cam_i = []
                #print(len(demo))
                for i, obs in enumerate(demo):
                    img_path = getattr(obs, "{}_rgb".format(cam_name))
                    img = np.array(Image.open(img_path)).transpose(2, 0, 1) ## img becomes of shape 3, h, w
                    cam_i.append(img)
                image_spec_task_i.append([cam_i[-1]])

                frame_idx = sample_frames(4, len(cam_i), self._sampling_strategy)
                cam_i = [cam_i[i] for i in frame_idx]
                video_spec_task_i.append(cam_i)

            video[ind,...] = np.asarray(video_spec_task_i).astype(np.float32)
            #print(np.asarray(image_spec_task_i).shape)
            #print(np.asarray(video_spec_task_i).shape)
            image_goal[ind,...] = np.asarray(image_spec_task_i).astype(np.float32)
        video = torch.from_numpy(video).type(torch.FloatTensor)
        video = video/255.0
        image_goal = torch.from_numpy(image_goal).type(torch.FloatTensor)
        image_goal = image_goal/255.0
        #print(image_goal.shape)
        #print(video.shape)
        return video, image_goal


    def update(self, step: int, replay_sample: dict) -> dict:
        robot_state = replay_sample['low_dim_state']
        observations = [
            replay_sample['%s_rgb' % self._camera_name],
            replay_sample['%s_point_cloud' % self._camera_name]
        ]
        if self._use_lang_goal:
            lang_goal_desc = replay_sample['lang_goal_desc']
            lang_goal_desc = self.batch_lang_goal_desc(lang_goal_desc)
            video_specification = None
            image_goal_specification = None
        if self._use_video_goal:
            video_specification, _ = self.get_visual_specification(replay_sample['task_name'], replay_sample['task_variation'])
            video_specification = video_specification.to(self._device)
            lang_goal_desc = None
            image_goal_specification = None
        if self._use_image_goal:
            _, image_goal_specification = self.get_visual_specification(replay_sample['task_name'], replay_sample['task_variation'])
            image_goal_specification = image_goal_specification.to(self._device)
            lang_goal_desc = None
            video_specification = None
        mu = self._actor(
                            observations=observations,
                            robot_state=robot_state,
                            lang_goal_desc=lang_goal_desc,
                            video_specification=video_specification,
                            image_goal_specification=image_goal_specification
                        )
        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        delta = F.mse_loss(
            mu, replay_sample['action'], reduction='none').mean(1)
        loss = (delta * loss_weights).mean()
        self._grad_step(loss, self._actor_optimizer,
                        self._actor.parameters(), self._grad_clip)
        self._summaries = {
            'pi/loss': loss,
            'pi/mu': mu.mean(),
        }
        return {'total_losses': loss}


    def _normalize_quat(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        task_name = np.asarray(observation.get('task_name'))
        task_variation = np.asarray(observation.get('task_variation'))

        observations = [
            observation['%s_rgb' % self._camera_name][0].to(self._device),
            observation['%s_point_cloud' % self._camera_name][0].to(self._device)
        ]
        robot_state = observation['low_dim_state'][0].to(self._device)

        if self._use_lang_goal:
            lang_goal_desc = observation.get('lang_goal_desc', None) # Bad behavior. We should get descriptions as observation
            lang_goal_desc = self.batch_lang_goal_desc(lang_goal_desc)
            video_specification = None
            image_goal_specification = None
        if self._use_video_goal:
            video_specification = self.get_visual_specification(task_name, task_variation)
            video_specification = video_specification.to(self._device)
            lang_goal_desc = None
            image_goal_specification = None
        if self._use_image_goal:
            _, image_goal_specification = self.get_visual_specification(task_name, task_variation)
            image_goal_specification = image_goal_specification.to(self._device)
            lang_goal_desc = None
            video_specification = None
        mu = self._actor(
                            observations=observations,
                            robot_state=robot_state,
                            lang_goal_desc=lang_goal_desc,
                            video_specification=video_specification,
                            image_goal_specification=image_goal_specification
                        )
        mu = torch.cat(
            [mu[:, :3], self._normalize_quat(mu[:, 3:7]), mu[:, 7:]], dim=-1)
        ignore_collisions = torch.Tensor([1.0]).to(mu.device)
        mu0 = torch.cat([mu[0], ignore_collisions])
        return ActResult(mu0.detach().cpu())

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))

        for tag, param in self._actor.named_parameters():
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (NAME, tag), param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (NAME, tag), param.data))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        self._actor.load_state_dict(
            torch.load(os.path.join(savedir, 'bc_actor.pt'),
                       map_location=torch.device('cpu')))
        print('Loaded weights from %s' % savedir)

    def save_weights(self, savedir: str):
        torch.save(self._actor.state_dict(),
                   os.path.join(savedir, 'bc_actor.pt'))
