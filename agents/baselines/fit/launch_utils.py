# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

import os
import logging
from typing import List
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer, ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import demo_loading_utils, utils
from helpers.custom_rlbench_env import CustomRLBenchEnv
from helpers.network_utils import ViT # Observation encoder
from helpers.preprocess_agent import PreprocessAgent

import torch
from torch.multiprocessing import Process, Value, Manager
import helpers.fit as fit
from helpers.fit import parse_config
from helpers.fit.fit import build_model#, batch_tokenize
# TODO: add FIT agent in network_utils/create your a separate one.
from helpers.fit_network_utils import FITAndFcsNet
from agents.baselines.fit.fit_agent import FITAgent

LOW_DIM_SIZE = 4


def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  image_size=[128, 128],
                  replay_size=3e5):
    lang_feat_dim = 1024

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    observation_elements.extend([
        ReplayElement('lang_goal_desc', (1,),
                      object), # Same as lang_goal
        ReplayElement('task', (),
                      str),
        ReplayElement('lang_goal', (1,),
                      object),  # language goal string for debugging and visualization
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(obs_tp1: Observation):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate([obs_tp1.gripper_pose[:3], quat,
                           [float(obs_tp1.gripper_open)]])


def _add_keypoints_to_replay(
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        description: str = '',
        device = 'cpu'):
    prev_action = None
    obs = inital_obs
    all_actions = []
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        action = _get_action(obs_tp1)
        all_actions.append(action)
        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) if terminal else 0

        obs_dict = utils.extract_obs(obs, t=k, prev_action=prev_action,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)
        del obs_dict['ignore_collisions']

        final_obs = {
            'task': task,
            'lang_goal': np.array([description], dtype=object), # Pass only language goal
            'lang_goal_desc': [description], # Pass only language goal
        }

        prev_action = np.copy(action)
        others = {'demo': True}
        others.update(final_obs)
        others.update(obs_dict)
        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1  # Set the next obs
    # Final step
    obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1, prev_action=prev_action,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)
    del obs_dict_tp1['ignore_collisions']
    # obs_dict_tp1['task'] = task
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)
    return all_actions


def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                device = 'cpu'):
    logging.getLogger().setLevel(cfg.framework.logging_level)

    logging.debug('Filling %s replay ...' % task)
    all_actions = []
    for d_idx in range(num_demos):
        # load demo from disk
        demo = rlbench_utils.get_stored_demos(
            amount=1, image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1, task_name=task,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=d_idx)[0]

        descs = demo._observations[0].misc['descriptions']

        # extract keypoints (a.k.a keyframes)
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo)

        if rank == 0:
            logging.info(f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints - {task}")

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue

            obs = demo[i]
            desc = descs[0]
            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            all_actions.extend(_add_keypoints_to_replay(
                cfg, task, replay, obs, demo, episode_keypoints, cameras,
                description=desc, device=device))
    logging.debug('Replay filled with demos.')
    return all_actions


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig,
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str]):
    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
                                        if torch.cuda.is_available() else 'cpu')
            p = Process(target=fill_replay,
                        args=(cfg,
                              obs_config,
                              rank,
                              replay,
                              task,
                              num_demos,
                              demo_augmentation,
                              demo_augmentation_every_n,
                              cameras,
                              model_device))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    logging.debug('Replay filled with multi demos.')


def create_agent(camera_name: str,
                 activation: str,
                 lr: float,
                 weight_decay: float,
                 image_resolution: list,
                 grad_clip: float,
                 norm = None):

    fit_model, tokenizer, visual_transform  = build_model(config_path=Path(os.path.join(fit.__path__[0], 'config.json')))

    vit = ViT(
        image_size=128,
        patch_size=8,
        num_classes=16,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=64,
        dropout=0.1,
        emb_dropout=0.1,
        channels=6,
    )

    actor_net = FITAndFcsNet(
        vit=vit,
        task_model=fit_model,
        tokenizer=tokenizer,
        visual_transform=visual_transform,
        input_resolution=image_resolution,
        filters=[64, 96, 128],
        kernel_sizes=[1, 1, 1],
        strides=[1, 1, 1],
        norm=norm,
        activation=activation,
        fc_layers=[128, 64, 3 + 4 + 1],
        low_dim_state_len=LOW_DIM_SIZE)
    #return actor_net

    bc_agent = FITAgent(
        actor_network=actor_net,
        camera_name=camera_name,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip)

    return PreprocessAgent(pose_agent=bc_agent)


if __name__ == '__main__':
    data = {}
    vid = torch.rand((4, 3, 256, 256))
    data["text"] = "Open the top drawer"
    data["video"] = vid.cuda()
    bs = 2
    lang_goal_desc = [data["text"] for _ in range(bs)]
    print(lang_goal_desc)
    robot_state = torch.rand((bs, 4)).to(device="cuda:0")
    observations = [
        torch.rand((bs,3,128,128)).to(device="cuda:0"),
        torch.rand((bs,3,128,128)).to(device="cuda:0")
    ]
    actor_network = create_agent(camera_name="bullshit",
                 activation="lrelu",
                 lr='1e-3',
                 weight_decay=1e-3,
                 image_resolution=84,
                 grad_clip=0.1,
                 norm = None)

    actor_network.build()
    actor_network.to(device="cuda:0")
    x = actor_network.forward(observations, robot_state, lang_goal_desc=lang_goal_desc)
    print(x.shape)
    exit()

    fit_model, tokenizer, vis_transform = agent
    print(fit_model)
    data["text"] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
    data["text"] = {key: val.cuda() for key, val in data['text'].items()}
    print(data['text'])
    vid = vis_transform(data["video"]) # Input has to be 4 dimensional, returns a vector of size [t, 3, 224, 224]
    vid = vid.unsqueeze(0)
    data["video"] = vid
    print(vid.shape)
    # Input: dict{'input_ids': [b, num_of_tokens], 'attention_mask': [b, num_tokens]}; Returns [b, 256] text feats
    text_feat = fit_model.module.compute_text(data["text"])
    # Input: [b, t, c, h, w]; Returns [b, 256] video feats
    vid_feat = fit_model.module.compute_video(data["video"])
    print(text_feat.shape, vid_feat.shape)

