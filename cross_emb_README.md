To generate retargeted data from a source robot to a target robot, run:

```bash
python metasim/scripts/collect_retarget_data.py --source_robot=franka --target_robot=sawyer --output_dir=retarget_data --batch_size 50 --num_demos 50 --max_timestep 200 --tasks CloseBox BasketballInHoop
```

This will load in the trajectory data from the source robot on the provided tasks, and for every state in that trajectory, retarget the target to have the same ee pose. Joint positions for both robots are stored in a pickle file

Batch_size: IKs to solve in a single step, this will always be done across the num_demos dimension
Max_timestep: Maximum timestep in a given trajectory to retarget


Then to render the paired images from the retargeted data, run:

```bash
python metasim/scripts/crossemb_render.py --data_path=retarget_data/franka_to_sawyer --task=CloseBox --robot=sawyer --num_envs 20 --num_steps 200 --sim isaaclab --random_level 0 --headless
python metasim/scripts/crossemb_render.py --data_path=retarget_data/franka_to_sawyer --task=CloseBox --robot=franka --num_envs 20 --num_steps 200 --sim isaaclab --random_level 0 --headless
```
This script has to be run seperately for each task and robot
--num_steps: Total number of images to render, these are sampled from random from all the joint poses stored above
--random_level: Randomization level to use for rendering, note level > 0 does not work for multiple envs and is non deterministic

These images will be saved to data_path/images

