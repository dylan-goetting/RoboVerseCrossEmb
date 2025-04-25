## Script for retargeting trajectories from a source robot to a target robot
## This script computes forward kinematics on the source robot and inverse kinematics on
## the target robot to make the end effectors in the same position relative to their root state

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
from copy import deepcopy
from dataclasses import dataclass, field

import rootutils
import tyro
from tqdm import tqdm

rootutils.setup_root(__file__, pythonpath=True)


@dataclass
class Args:
    source_robot: str = "franka"
    """Source robot name from which to retarget"""
    target_robot: str = "sawyer"
    """Target robot name to retarget to"""
    tasks: list[str] = field(default_factory=lambda: ["CloseBox"])
    """List of task names to retarget"""
    noise_std: float = 0.0
    """Standard deviation of noise to add to source robot joint positions (0 for no noise)"""
    output_dir: str = "retarget_data"
    """Directory to save retargeted data"""
    device: str = "cuda:0"
    """Device to run computation on, e.g. 'cuda:0', 'cuda:1', 'cpu'"""
    batch_size: int = 40
    """Batch size for retargeting"""
    max_timestep: int = 200
    """Maximum number of timesteps to retarget"""
    num_demos: int = 40
    """Number of demonstrations to retarget"""

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


#########################################
### Import packages
#########################################
import torch

from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_task


def retarget_states(states, source_robot, target_robot, noise_std=0.01, device="cuda:0"):
    """Retarget a batch of states from a source robot to a target robot.

    Args:
        states: A list of dictionaries containing robot state information
        source_robot: The source robot configuration
        target_robot: The target robot configuration
        noise_std: Standard deviation of noise to add to the source robot joint positions (0 for no noise)
        device: Device to run computation on, e.g. 'cuda:0', 'cuda:1', 'cpu'

    Returns:
        List of retargeted states with target robot joint positions
    """
    from curobo.types.math import Pose

    from metasim.utils.kinematics_utils import get_curobo_models

    # Load models and FK/IK functions
    _, _, target_ik = get_curobo_models(target_robot)
    _, source_fk, _ = get_curobo_models(source_robot)

    # Get joint ordering
    source_joint_names = list(source_robot.joint_limits.keys())
    target_joint_names = list(target_robot.joint_limits.keys())

    # Extract source positions and orientations
    source_pos = [state["robots"][source_robot.name]["pos"] for state in states]
    source_rot = [state["robots"][source_robot.name]["rot"] for state in states]

    # Convert joint positions dictionaries to tensor - shape (batch_size, num_joints)
    q_tensor = torch.stack([
        torch.tensor([state["robots"][source_robot.name]["dof_pos"][joint_name] for joint_name in source_joint_names])
        for state in states
    ]).cuda()

    # Add noise if specified
    if noise_std > 0:
        noise = torch.randn_like(q_tensor) * noise_std
        noisy_q_tensor = q_tensor + noise

        # Store the noisy joint positions in each state if needed
        for batch_idx, state in enumerate(states):
            state["robots"][source_robot.name]["dof_pos"] = {
                joint_name: noisy_q_tensor[batch_idx, joint_idx].item()
                for joint_idx, joint_name in enumerate(source_joint_names)
            }

        # Use noisy joint positions for FK
        q_tensor = noisy_q_tensor

    # Get gripper state
    source_gripper_q = q_tensor[:, -len(source_robot.gripper_release_q) :]
    gripper_actuate_tensor = torch.tensor(source_robot.gripper_actuate_q).cuda()
    gripper_release_tensor = torch.tensor(source_robot.gripper_release_q).cuda()

    is_actuated = torch.norm(source_gripper_q - gripper_actuate_tensor, dim=1) < torch.norm(
        source_gripper_q - gripper_release_tensor, dim=1
    )

    # Compute FK on source robot - already on correct device from tensor creation
    ee_positions, ee_quaternions = source_fk(q_tensor)

    # Do IK on target robot
    target_pose = Pose(ee_positions, ee_quaternions)
    ik_result = target_ik.solve_batch(target_pose, num_seeds=5)
    if not all(ik_result.success):
        log.warning("IK failed for some states")

    # Extract retargeted joint positions - already on correct device
    retargeted_q = ik_result.solution.squeeze(1)

    # Create gripper tensors for target robot
    target_actuate_tensor = torch.tensor(target_robot.gripper_actuate_q).cuda()
    target_release_tensor = torch.tensor(target_robot.gripper_release_q).cuda()

    # Stack the appropriate gripper state based on actuation
    retargeted_gripper_q = torch.stack(
        [target_actuate_tensor if is_actuated[i] else target_release_tensor for i in range(len(is_actuated))],
        dim=0,
    )
    retargeted_q = torch.cat([retargeted_q, retargeted_gripper_q], dim=1)

    # Create retargeted states
    retargeted_states = []
    for i, state in enumerate(states):
        retargeted_state = deepcopy(state)
        retargeted_state["robots"][target_robot.name] = {
            "dof_pos": {
                joint_name: retargeted_q[i, index].item() for index, joint_name in enumerate(target_joint_names)
            },
            "pos": source_pos[i],
            "rot": source_rot[i],
        }
        retargeted_states.append(retargeted_state)

    return retargeted_states


def retarget_trajectories(
    all_demos_states,
    source_robot,
    target_robot,
    noise_std=0.01,
    batch_size=50,
    device="cuda:0",
    max_timestep=20,
):
    """Retarget all states in multiple trajectories from a source robot to a target robot.

    Args:
        all_demos_states: A list of lists of dictionaries containing trajectory data for multiple demos
        source_robot: The source robot configuration
        target_robot: The target robot configuration
        add_noise: Whether to add noise to the source robot joint positions
        noise_std: Standard deviation of the noise
        batch_size: Number of trajectories to process in a single batch
        device: Device to run computation on, e.g. 'cuda:0', 'cuda:1', 'cpu'

    Returns:
        List of retargeted trajectories with target robot joint positions
    """
    num_demos = len(all_demos_states)
    max_timesteps = min(max(len(demo) for demo in all_demos_states), max_timestep)

    # Pre-allocate output trajectories
    retargeted_trajectories = [[] for _ in range(num_demos)]

    # Process each timestep across all demonstrations in batches
    for timestep in tqdm(range(max_timesteps), desc="Retargeting timesteps across trajectories"):
        # For each timestep, collect states from all demos (if available)
        timestep_states = []
        demo_indices = []

        for demo_idx, demo_states in enumerate(all_demos_states):
            if timestep < len(demo_states):
                timestep_states.append(demo_states[timestep])
                demo_indices.append(demo_idx)

        # Process in batches
        for batch_start in range(0, len(timestep_states), batch_size):
            batch_end = min(batch_start + batch_size, len(timestep_states))
            batch_states = timestep_states[batch_start:batch_end]
            batch_indices = demo_indices[batch_start:batch_end]

            # Retarget the batch
            retargeted_batch = retarget_states(
                batch_states, source_robot, target_robot, noise_std=noise_std, device=device
            )

            # Store the retargeted states in the correct trajectories
            for batch_idx, demo_idx in enumerate(batch_indices):
                retargeted_trajectories[demo_idx].append(retargeted_batch[batch_idx])

    return retargeted_trajectories


def save_retargeted_data(trajectories, task_name, source_robot_name, target_robot_name, save_dir):
    """Save all retargeted trajectory data together.

    Args:
        trajectories: List of retargeted trajectories
        task_name: Name of the task
        source_robot_name: Name of the source robot
        target_robot_name: Name of the target robot
        save_dir: Directory to save the retargeted data
    """
    # Create directory structure if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Add metadata
    metadata = {
        "task": task_name,
        "source_robot": source_robot_name,
        "target_robot": target_robot_name,
        "num_trajectories": len(trajectories),
    }

    # Create combined data structure with metadata and all trajectories
    retargeted_data = {"metadata": metadata, "trajectories": trajectories}

    # Save as a demo using the save_demo utility
    import pickle as pkl

    with open(os.path.join(save_dir, "retargeted_data.pkl"), "wb") as f:
        pkl.dump(retargeted_data, f)
    log.info(f"Saved {len(trajectories)} retargeted trajectories to {save_dir}")


def main():
    """Main function to process and retarget trajectories."""
    # Use specified device
    device = args.device
    log.info(f"Using device: {device}")

    source_robot = get_robot(args.source_robot)
    target_robot = get_robot(args.target_robot)

    log.info(f"Retargeting from {args.source_robot} to {args.target_robot}")

    # Process each task
    for task_name in args.tasks:
        log.info(f"Processing task: {task_name}")
        task = get_task(task_name)

        # Get trajectory for source robot (passing None for handler as we don't need simulation)
        log.info(f"Loading trajectory data for task {task_name} with robot {args.source_robot}")
        init_states, all_actions, all_states = get_traj(task, source_robot, None)

        # Process all demonstrations at once using batched retargeting
        num_demos = min(len(all_states), args.num_demos)
        log.info(f"Retargeting {num_demos} demonstrations for task {task_name}")
        retargeted_trajectories = retarget_trajectories(
            all_states[:num_demos],
            source_robot,
            target_robot,
            noise_std=args.noise_std,
            batch_size=args.batch_size,
            device=device,
            max_timestep=args.max_timestep,
        )

        # Save all retargeted trajectories together
        log.info(f"Saving all retargeted trajectories for task {task_name}")
        save_dir = os.path.join(args.output_dir, f"{args.source_robot}_to_{args.target_robot}", task_name)

        save_retargeted_data(retargeted_trajectories, task_name, args.source_robot, args.target_robot, save_dir)

    log.info("Retargeting complete")


if __name__ == "__main__":
    main()
