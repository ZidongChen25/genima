
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from env.rlbench import GenimaRLBenchEnv, ActionModeType
from env.rlbench_utils import _get_action_mode
from robobase.envs.wrappers.action_sequence import ActionSequence
from gymnasium.wrappers import TimeLimit

@hydra.main(config_path="../robobase/robobase/cfgs", config_name="env/rlbench", version_base=None)
def main(cfg: DictConfig):
    # Override config values to match user request
    OmegaConf.set_struct(cfg, False)
    cfg.env.train_tasks = ["lamp_on"]
    cfg.env.episode_length = 150
    # Update dataset root
    cfg.env.dataset_root = "/home/zc1525/Desktop/genima/data/train_data/"
    
    print(f"Loading task: {cfg.env.train_tasks[0]}")
    
    # Create Env
    from rlbench.observation_config import ObservationConfig
    from robobase.envs.rlbench import TASK_TO_LOW_DIM_SIM
    
    # Monkey-patch TASK_TO_LOW_DIM_SIM
    TASK_TO_LOW_DIM_SIM["lamp_on"] = 0
    
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera.rgb = True
    obs_config.wrist_camera.rgb = True
    obs_config.left_shoulder_camera.rgb = True
    obs_config.right_shoulder_camera.rgb = True
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    
    action_mode = _get_action_mode(ActionModeType.JOINT_POSITION)

    # We need to instantiate the env to get demos, but we don't need to wrap it yet
    env = GenimaRLBenchEnv(
        task_name=cfg.env.train_tasks[0],
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=cfg.env.dataset_root,
        headless=True,
    )
    
    print("Launching env...")
    env._launch()
    
    print("Loading demos...")
    # Load 1 demo
    demos = env.get_demos(1, desc="inspect_padding")
    
    print(f"Loaded {len(demos)} demos.")
    print(f"Demo length: {len(demos[0])} steps.")
    
    # Create DemoEnv
    from robobase.envs.env import DemoEnv
    demo_env = DemoEnv(demos, env.action_space, env.observation_space)
    
    # Wrap DemoEnv
    # 1. TimeLimit
    demo_env = TimeLimit(demo_env, max_episode_steps=cfg.env.episode_length)
    
    # 2. ActionSequence
    sequence_length = 20
    demo_env = ActionSequence(demo_env, sequence_length=sequence_length)
    
    # Reset
    obs, info = demo_env.reset()
    print("Reset done.")
    
    # Iterate
    chunk_idx = 0
    done = False
    while not done:
        # For DemoEnv, action doesn't matter, it pops from demo
        action = demo_env.action_space.sample() 
        obs, reward, term, trunc, info = demo_env.step(action)
        
        print(f"Chunk {chunk_idx} (Steps {chunk_idx*sequence_length}-{(chunk_idx+1)*sequence_length}):")
        if "action_sequence_mask" in info:
            mask = info["action_sequence_mask"]
            print(f"  Mask: {mask}")
            print(f"  Mask Shape: {mask.shape}")
            if not np.all(mask):
                print("  -> PADDING DETECTED!")
        else:
            print("  Mask not found!")
            
        chunk_idx += 1
        if term or trunc:
            done = True
            print("Episode done.")
            
if __name__ == "__main__":
    main()
