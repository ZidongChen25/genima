
import numpy as np
from robobase.utils import observations_to_timesteps, DemoStep
from gymnasium import spaces

class DummyDemoStep:
    def __init__(self, gripper_open):
        self.gripper_open = gripper_open
        self.joint_positions = np.zeros(7)
        self.gripper_matrix = np.eye(4)
        self.misc = {"joint_position_action": np.zeros(8)}

def test_demo_loading():
    # Create a dummy demo with 50 steps
    demo = [DummyDemoStep(1.0) for _ in range(50)]
    
    action_space = spaces.Box(low=-1, high=1, shape=(8,))
    
    # Mock action function
    def mock_act_func(curr, next, space):
        return np.zeros(8, dtype=np.float32)
        
    loaded_demos = observations_to_timesteps(
        demo, 
        action_space, 
        skipping=False, 
        obs_to_act_func=mock_act_func
    )
    
    print(f"Loaded {len(loaded_demos)} demos")
    for d_idx, d in enumerate(loaded_demos):
        print(f"Demo {d_idx} length: {len(d)}")
        # Check the last step
        last_step = d[-1]
        # (next_demo_step, r, done, False, info)
        print(f"Last step done flag: {last_step[2]}")
        
        if last_step[2]:
            print("SUCCESS: Last step has done=True")
        else:
            print("FAILURE: Last step has done=False")

if __name__ == "__main__":
    test_demo_loading()
