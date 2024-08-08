"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""
import os
import time
import json
import random
import imageio
import mimicgen

import PIL
import torch
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

from mimicgen.configs import config_factory, MG_TaskSpec
from mimicgen.generate import generate as generate_configs, generate_instruction
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.env_interfaces.base import make_interface, MG_EnvInterface

from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robosuite.environments.robot_env import RobotEnv

import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class MGTrainingTask:
    config: dict
    env: RobotEnv
    env_interface: MG_EnvInterface
    data_generator: DataGenerator
    num_episodes: int = 0
    
    
class MGStreamingDataset(IterableDataset):
    """
    Streaming dataset simulator using MimicGen (mimicgen.github.io/)
    """
    def __init__(self, 
        tasks: List[str],           # tasks to choose between
        task_weights: List[float],  # task selection weights
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        resize_resolution: Tuple[int, int],
        shuffle = True,
        image_aug: bool = False,
        camera: str = 'agentview',  # camera name to use (agentview)
        save_dir: str = None,       # directory to save sample data
        save_freq: int = None,      # save every N episodes
    ) -> None:
        """
        PyTorch dataset connector for live MimicGen training examples.
        """
        if not tasks:
            raise ValueError("requires at least one mimicgen task / environment")
            
        if not task_weights:
            task_weights = [1.0/len(tasks)] * len(tasks)
          
        self.tasks = []
        self.task_weights = task_weights

        self.camera = camera
        self.shuffle = shuffle
        self.image_aug = image_aug
 
        self.frames = 0
        self.episodes = 0
        
        self.save_dir = save_dir
        self.save_freq = save_freq
        
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "mimicgen": {
                "action": {
                    "q01": np.array([-1.0] * 6 + [0.0], dtype=np.float32), 
                    "q99": np.array([1.0] * 7, dtype=np.float32),
                    "mask": np.array([True] * 6 + [False], dtype=bool),
                 }
            }
        }
        
        if not save_dir:
            save_dir = "/tmp/mimicgen"
            
        self.save_dir = os.path.join(save_dir, f"rank={torch.distributed.get_rank()}+pid={os.getpid()}")

        generate_configs(
            tasks, episodes=1e9, 
            output=self.save_dir, 
            cameras=[camera], 
            camera_width=resize_resolution[-1], 
            camera_height=resize_resolution[0],
            simulate=True,
        )

        for task in tasks:
            self.tasks.append(self.init_env(task))
            
    def init_env(self, task):  
        """
        Create simulator environment and data generator for task
        """
        task_name, task_level = task.split('_')
        config_path = os.path.join(self.save_dir, 'config', f"demo_src_{task_name.lower()}_task_{task_level.upper()}.json")
        
        with open(config_path, "r") as f:
            ext_cfg = json.load(f)
            
            if "meta" in ext_cfg:
                del ext_cfg["meta"] # robomimic generates this part of config, unused by MimicGen
                
        mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])

        # update config with external json - this will throw errors if
        # the external config has keys not present in the base config
        with mg_config.values_unlocked():
            mg_config.update(ext_cfg)

            # We assume that the external config specifies all subtasks, so
            # delete any subtasks not in the external config.
            source_subtasks = set(mg_config.task.task_spec.keys())
            new_subtasks = set(ext_cfg["task"]["task_spec"].keys())
            for subtask in (source_subtasks - new_subtasks):
                print("deleting subtask {} in original config".format(subtask))
                del mg_config.task.task_spec[subtask]

        # path to source dataset
        source_dataset_path = os.path.expandvars(os.path.expanduser(mg_config.experiment.source.dataset_path))

        # get environment metadata from dataset
        env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)

        # set seed for generation TODO augment by PID?
        random.seed(mg_config.experiment.seed + torch.distributed.get_rank())
        np.random.seed(mg_config.experiment.seed + torch.distributed.get_rank())

        # get list of source demonstration keys from source hdf5
        all_demos = MG_FileUtils.get_all_demos_from_dataset(
            dataset_path=source_dataset_path,
            filter_key=mg_config.experiment.source.filter_key,
            start=mg_config.experiment.source.start,
            n=mg_config.experiment.source.n,
        )

        # simulation environment
        env = RobomimicUtils.create_env(
            env_meta=env_meta,
            env_class=None,
            env_name=mg_config.experiment.task.name,
            robot=mg_config.experiment.task.robot,
            gripper=mg_config.experiment.task.gripper,
            camera_names=[self.camera],
            camera_height=mg_config.obs.camera_height,
            camera_width=mg_config.obs.camera_width,
            render=False, 
            render_offscreen=True,
            use_image_obs=True,
            use_depth_obs=False,
        )

        # get information necessary to create env interface
        env_interface_name, env_interface_type = MG_FileUtils.get_env_interface_info_from_dataset(
            dataset_path=source_dataset_path,
            demo_keys=all_demos,
        )
        # possibly override from config
        if mg_config.experiment.task.interface is not None:
            env_interface_name = mg_config.experiment.task.interface
        if mg_config.experiment.task.interface_type is not None:
            env_interface_type = mg_config.experiment.task.interface_type

        # create environment interface to use during data generation
        env_interface = make_interface(
            name=env_interface_name,
            interface_type=env_interface_type,
            # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
            env=env.base_env,
        )

        # get task spec object from config
        task_spec_json_string = mg_config.task.task_spec.dump()
        task_spec = MG_TaskSpec.from_json(json_string=task_spec_json_string)

        # make data generator object
        data_generator = DataGenerator(
            task_spec=task_spec,
            dataset_path=source_dataset_path,
            demo_keys=all_demos,
        )

        print(f"\n==== MimicGen Environment (rank={torch.distributed.get_rank()}  PID={os.getpid()}) ====\n")
        print(json.dumps(env.serialize(), indent=4))
        print(f"\nEnvironment Interface:\n\n{env_interface}")
        print(f"\nData Generator:\n\n{data_generator}\n")

        # add task state to list of tasks
        return MGTrainingTask(
            config=mg_config, env=env, 
            env_interface=env_interface,
            data_generator=data_generator
        )
            
    def generate(self, task):
        try:
            return task.data_generator.generate(
                env=task.env,
                env_interface=task.env_interface,
                select_src_per_subtask=task.config.experiment.generation.select_src_per_subtask,
                transform_first_robot_pose=task.config.experiment.generation.transform_first_robot_pose,
                interpolate_from_last_target_pose=task.config.experiment.generation.interpolate_from_last_target_pose,
                render=False, # disable on-screen
            )
        except env.rollout_exceptions as e:
            print(f"\n==== WARNING - MimicGen {task.config.name} roll-out exception:  {e}   (rank={torch.distributed.get_rank()}  PID={os.getpid()})\n")
    
    def transform(self, episode, frame=None):
        if frame is None:
            return [
                self.transform(episode, frame=n)
                for n in range(episode['actions'].shape[0])
            ]
        
        image = PIL.Image.fromarray(episode['observations'][frame][self.camera + '_image'])
        action = episode['actions'][frame]
        
        # invert gripper from [-1,1] to [0,1]
        action[-1] = 1.0 - ((action[-1] + 1.0) * 0.5) 

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {episode['instruction']}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)    
                    
    def __iter__(self) -> Dict[str, Any]:
        task_choices = []
        
        while True:
            if not task_choices:
                if self.shuffle:
                    task_choices = random.choices(range(len(self.tasks)), weights=self.task_weights, k=1000)
                else:
                    task_choices = list(range(len(self.tasks)))
                    
            time_begin = time.perf_counter()
            task = self.tasks[task_choices.pop()]
            episode = self.generate(task)
            time_elapsed = time.perf_counter() - time_begin
            
            if not episode:
                continue

            success = bool(episode['success'])
            frames = episode['actions'].shape[0] if success else 0
            
            print(f"MimicGen | rank={torch.distributed.get_rank()}  PID={os.getpid()}  task={task.config.name}  time={time_elapsed:.3f}  episode={self.episodes}  frames={frames}  success={success}")
            
            if not success:
                continue
            
            episode['instruction'] = generate_instruction(task.config.name, task.env, shuffle=task.num_episodes)
            
            if self.save_freq and self.episodes % self.save_freq == 0 or self.episodes < 3:
                video_file = os.path.join(self.save_dir, f"{self.episodes}.mp4")
                os.makedirs(self.save_dir, exist_ok=True)
                images = [x[self.camera + '_image'] for x in episode['observations']]
                imageio.mimsave(video_file, np.stack(images), fps=20)
                print(f"MimicGen | saved episode {self.episodes} to {video_file}")
                with open(os.path.join(self.save_dir, f"{self.episodes}.json"), 'w') as f:
                    json.dump({
                        'instruction': episode['instruction'],
                        'actions': episode['actions'].tolist(),
                    }, f, indent=2)

            if self.shuffle:
                frame_choices = random.sample(range(frames), k=frames)
            else:
                frame_choices = range(frames)
                
            for n in frame_choices:
                yield self.transform(episode, frame=n)
                self.frames += 1
                
            task.num_episodes += 1
            self.episodes += 1

    #def __len__(self) -> int:
    #    return None  # streaming (unlimited length)

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

