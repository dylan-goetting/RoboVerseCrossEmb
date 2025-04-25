from __future__ import annotations

#########################################
## Setup logging
#########################################
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


#########################################
### Add command line arguments
#########################################
import os
from dataclasses import dataclass

import rootutils
import tyro

rootutils.setup_root(__file__, pythonpath=True)


@dataclass
class Args:
    data_path: str = "retarget_data/franka_to_sawyer"
    """Path to the retargeting data output by collect_retarget_data.py"""
    task: str = "CloseBox"
    """Name of the task"""
    robot: str = "franka"
    """Name of the robot"""
    num_envs: int = 4
    """Number of parallel environments, find a proper number for best performance on your machine"""
    num_steps: int = 20
    """Number of steps to render across all trajectories"""
    sim: str = "isaaclab"
    """Simulator backend"""
    random_level: int = 0
    """Randomization level"""
    headless: bool = True
    """Whether to run in headless mode"""
    random_seed: int = 42
    """Random seed for reproducibility"""

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)

#########################################
### Import packages
#########################################
import pickle
import random

import numpy as np

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_env_class

np.random.seed(args.random_seed)
random.seed(args.random_seed)


def load_retarget_data(retarget_data_path, task):
    """Load retargeted data from the specified path.

    Args:
        retarget_data_path: Path to the retargeting data

    Returns:
        Dict containing retargeted data
    """
    pkl_path = os.path.join(retarget_data_path, task, "retargeted_data.pkl")
    with open(pkl_path, "rb") as f:
        retarget_data = pickle.load(f)
    log.info(f"Loaded {len(retarget_data['trajectories'])} retargeted trajectories")
    return retarget_data


def render_trajectories(trajectories, task, robot, num_envs, num_steps, output_dir, random_level):
    """Render images from robot trajectories.

    Args:
        trajectories: List of retargeted trajectories
        task: Task configuration
        robot: Robot configuration
        num_envs: Number of environments to run in parallel
        num_steps: Total number of steps to render across all trajectories
        output_dir: Directory to save rendered images
    """
    # Initialize environment
    handler_class = get_sim_env_class(SimType(args.sim))
    camera = PinholeCameraCfg(data_types=["rgb", "depth"], pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
    camera1 = PinholeCameraCfg(data_types=["rgb", "depth"], pos=(1.7, 0.0, 1.3), look_at=(0.0, 0.0, 0.0))
    # camera2 = PinholeCameraCfg(data_types=["rgb", "depth"], pos=(1.3, 0.0, 1.7), look_at=(0.0, 0.0, 0.0))
    # camera3 = PinholeCameraCfg(data_types=["rgb", "depth"], pos=(1, 0.0, 2), look_at=(0.0, 0.0, 0.0))
    cameras = [camera]
    # Configure scene
    scenario = ScenarioCfg(
        task=task,
        robot=robot,
        scene=None,
        cameras=cameras,
        random=RandomizationCfg(level=random_level),
        try_add_table=True,
        render=RenderCfg(),
        split="all",
        sim=args.sim,
        headless=args.headless,
        num_envs=num_envs,
    )

    # Create environment

    env = handler_class(scenario)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Flatten trajectories to get the first num_steps frames across all trajectories
    states = []
    for traj in trajectories:
        states += traj

    # Limit to num_steps
    states = random.sample(states, num_steps)
    # rng_state = random.getstate()

    log.info(f"Rendering {len(states)} steps from {robot} trajectories")

    # Render in batches for better performance
    for batch_start in range(0, len(states), num_envs):
        batch_end = min(batch_start + num_envs, len(states))
        batch_size = batch_end - batch_start

        # Extract the states for this batch
        batch_states = states[batch_start:batch_end]

        # Pad the batch if necessary
        if batch_size < num_envs:
            # Use the last state to pad the batch
            padding = [states[-1]] * (num_envs - batch_size)
            padded_states = batch_states + padding
        else:
            padded_states = batch_states

        # Reset environment with the batch of states
        # random.setstate(rng_state)
        obs, _ = env.reset(states=padded_states)

        # Extract RGB images from observations and save only the valid ones
        for i in range(batch_size):
            global_step_idx = batch_start + i
            # Save the image directly
            for cameraNdx in range(len(cameras)):
                img = obs.cameras["camera0"].rgb[i].cpu().numpy()
                filename = f"{robot}_task{task}_step{global_step_idx}_camera{cameraNdx}.png"
                filepath = os.path.join(output_dir, filename)
                save_single_image(img, filepath)

        log.info(f"Processed batch {batch_start // num_envs + 1}/{(len(states) - 1) // num_envs + 1}")

    # Clean up
    log.info("Closing environment")
    env.close()
    log.info("Environment closed")


def save_single_image(img_data, filepath):
    """Save a single image to a file.

    Args:
        img_data: Image data as a numpy array
        filepath: Path to save the image to

    """
    import cv2

    if img_data.dtype != np.uint8:
        if img_data.max() <= 1.0:
            img_data = (img_data * 255).astype(np.uint8)
        else:
            img_data = img_data.astype(np.uint8)

    # OpenCV expects BGR order
    if img_data.shape[-1] == 3:  # If it has RGB channels
        img_data_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_data_bgr)
    else:
        cv2.imwrite(filepath, img_data)

    log.info(f"Saved image to {filepath}")


def main():
    """Main function to render and save trajectory images."""
    # Get the directory of the data path for output
    output_dir = args.data_path + "/images"
    os.makedirs(output_dir, exist_ok=True)

    # Load retargeting data
    retarget_data = load_retarget_data(args.data_path, args.task)

    # Render trajectories and save images
    render_trajectories(
        retarget_data["trajectories"],
        args.task,
        args.robot,
        args.num_envs,
        args.num_steps,
        output_dir,
        args.random_level,
    )

    log.info("Rendering complete!")


if __name__ == "__main__":
    main()
