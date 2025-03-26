from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio.v2 as iio
import numpy as np
import rootutils
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.cfg.randomization import RandomizationMetaCfg
from metasim.cfg.scenario import ScenarioMetaCfg
from metasim.cfg.sensors.cameras import PinholeCameraMetaCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
from roboverse_learn.algorithms import PolicyRunner, get_runner


@dataclass
class Args:
    random: RandomizationMetaCfg
    """Domain randomization options"""
    task: str
    """Task name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym"] = "isaaclab"
    """Simulator backend"""
    max_demo: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    headless: bool = False
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    task_id_range_low: int = 0
    """Low end of the task id range"""
    task_id_range_high: int = 1000
    """High end of the task id range"""
    checkpoint_path: str = ""
    """Path to the checkpoint"""
    algo: str = "diffusion_policy"
    """Algorithm to use"""
    subset: str = "pickcube_l0"
    """Subset your ckpt trained on"""
    action_set_steps: int = 2
    """Number of steps to take for each action set"""
    save_video_freq: int = 1
    """Frequency of saving videos"""
    max_step: int = 250
    """Maximum number of steps to collect"""
    gpu_id: int = 0
    """GPU ID to use"""
    target_robot: str = "sawyer"
    """Robot name"""
    source_robot: str = "franka"
    """Source robot name"""

    def __post_init__(self):
        if self.random.table and not self.table:
            log.warning("Cannot enable table randomization without a table, disabling table randomization")
            self.random.table = False
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def retarget_initial_state(initial_states, source_robot, target_robot):
    import torch
    from curobo.types.math import Pose

    from metasim.utils.kinematics_utils import get_curobo_models

    # Load models and FK functions
    _, _, target_ik = get_curobo_models(target_robot)
    _, source_fk, _ = get_curobo_models(source_robot)

    # Get joint ordering for the source robot
    source_joint_names = source_robot.joint_limits.keys()  # ordered list
    target_joint_names = target_robot.joint_limits.keys()  # ordered list
    num_states = len(initial_states)
    source_pos = [state[source_robot.name]["pos"] for state in initial_states]
    source_rot = [state[source_robot.name]["rot"] for state in initial_states]
    # Convert list of dicts to tensor of shape [N, num_joints]
    q_tensor = torch.stack([
        torch.tensor([state[source_robot.name]["dof_pos"][joint_name] for joint_name in source_joint_names])
        for state in initial_states
    ])  # shape: (batch_size, num_joints)
    source_gripper_q = q_tensor[:, -len(source_robot.gripper_release_q) :]
    is_actuated = torch.norm(
        source_gripper_q - torch.tensor(source_robot.gripper_actuate_q).to(source_gripper_q.device), dim=1
    ) < torch.norm(source_gripper_q - torch.tensor(source_robot.gripper_release_q).to(source_gripper_q.device), dim=1)
    ee_positions, ee_quaternions = source_fk(q_tensor.cuda())

    # do IK on target robot
    target_pose = Pose(ee_positions, ee_quaternions)
    ik_result = target_ik.solve_batch(target_pose, num_seeds=5)
    if not all(ik_result.success):
        print("IK failed")
        import pdb

        pdb.set_trace()
        # raise ValueError("IK failed")
    else:
        print("IK succeeded")
    # Extract the retargeted joint positions
    retargeted_q = ik_result.solution.squeeze(1)
    retargeted_gripper_q = torch.stack(
        [
            torch.tensor(target_robot.gripper_actuate_q if is_actuated[i] else target_robot.gripper_release_q).to(
                retargeted_q.device
            )
            for i in range(len(is_actuated))
        ],
        dim=0,
    )
    retargeted_q = torch.cat([retargeted_q, retargeted_gripper_q], dim=1)
    # Convert tensor to list of dictionaries for each state
    for i in range(num_states):
        del initial_states[i][source_robot.name]
        initial_states[i][target_robot.name] = {
            "dof_pos": {joint_name: retargeted_q[i, index] for index, joint_name in enumerate(target_joint_names)},
            "pos": source_pos[i],
            "rot": source_rot[i],
        }
    return initial_states


def main():
    num_envs: int = args.num_envs
    log.info(f"Using GPU device: {args.gpu_id}")

    task = get_task(args.task)
    task.episode_length = args.action_set_steps * args.max_step
    target_robot = get_robot(args.target_robot)
    source_robot = get_robot(args.source_robot)

    camera = PinholeCameraMetaCfg(pos=(1.5, 0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioMetaCfg(
        task=args.task,
        robot=args.target_robot,
        cameras=[camera],
        random=args.random,
        sim=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    tic = time.time()
    env_class = get_sim_env_class(SimType(scenario.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        ckpt_name = (
            args.checkpoint_path.split("/")[-3]
            + "_"
            + args.checkpoint_path.split("/")[-1].split(".")[0]
            + "_ActStep"
            + str(args.action_set_steps)
            + "_"
            + time_str
        )
    except:
        ckpt_name = (
            args.checkpoint_path.split("/")[-1].split(".")[0] + "_ActStep" + str(args.action_set_steps) + "_" + time_str
        )
    ckpt_name = f"{args.task}/{args.target_robot}/{args.algo}_{ckpt_name}"
    runnerCls = get_runner(args.algo)
    policyRunner: PolicyRunner = runnerCls(
        scenario=scenario,
        num_envs=num_envs,
        checkpoint_path=args.checkpoint_path,
        device=f"cuda:{args.gpu_id}",
        task_name=args.task,
        subset=args.subset,
    )

    ## Data
    tic = time.time()
    assert os.path.exists(task.traj_filepath), f"Trajectory file: {task.traj_filepath} does not exist."
    # if source_robot:
    source_init_states, all_actions, all_states = get_traj(task, source_robot, env.handler)
    init_states = retarget_initial_state(source_init_states.copy(), source_robot, target_robot)
    # else:
    #     init_states, all_actions, all_states = get_traj(task, robot, env.handler)
    num_demos = len(init_states)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    total_success = 0
    total_completed = 0
    if args.max_demo is None:
        max_demos = args.task_id_range_high - args.task_id_range_low
    else:
        max_demos = args.max_demo
    max_demos = min(max_demos, num_demos)
    for demo_start_idx in range(args.task_id_range_low, args.task_id_range_low + max_demos, num_envs):
        demo_end_idx = min(demo_start_idx + num_envs, num_demos)

        ## Reset before first step
        tic = time.time()
        obs, extras = env.reset(states=init_states[demo_start_idx:demo_end_idx])
        policyRunner.reset()
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")

        step = 0
        MaxStep = args.max_step
        SuccessOnce = [False] * num_envs
        TimeOut = [False] * num_envs
        images_list = []

        while step < MaxStep:
            log.debug(f"Step {step}")

            images_list.append(np.array(obs["rgb"]))
            action = policyRunner.get_action(obs)

            for round_i in range(args.action_set_steps):
                obs, reward, success, time_out, extras = env.step(action)
            env.handler.render()

            # eval
            SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
            TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
            step += 1
            if all(SuccessOnce):
                break

        SuccessEnd = success.tolist()
        total_success += SuccessOnce.count(True)
        total_completed += len(SuccessOnce)
        os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
        for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
            demo_idx_str = str(demo_idx).zfill(4)
            if i % args.save_video_freq == 0:
                iio.mimwrite(f"tmp/{ckpt_name}/{demo_idx}.mp4", [images[i] for images in images_list])
            with open(f"tmp/{ckpt_name}/{demo_idx_str}.txt", "w") as f:
                f.write(f"Demo Index: {demo_idx}\n")
                f.write(f"Num Envs: {num_envs}\n")
                f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                f.write(f"TimeOut: {TimeOut[i]}\n")
                f.write(f"Cumulative Average Success Rate: {total_success / total_completed}\n")
        log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
        log.info("Num Envs: ", num_envs)
        log.info(f"SuccessOnce: {SuccessOnce}")
        log.info(f"SuccessEnd: {SuccessEnd}")
        log.info(f"TimeOut: {TimeOut}")
    log.info(f"FINAL RESULTS: {total_success / total_completed}")
    with open(f"tmp/{ckpt_name}/final_stats.txt", "w") as f:
        f.write(f"Total Success: {total_success}\n")
        f.write(f"Total Completed: {total_completed}\n")
        f.write(f"Average Success Rate: {total_success / total_completed}\n")
    env.close()


if __name__ == "__main__":
    main()
