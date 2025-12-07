import gymnasium as gym
import numpy as np

d = gym.spaces.Dict({"a": gym.spaces.Discrete(1)})
print(f"'a' in d: {'a' in d}")
print(f"'a' in d.spaces: {'a' in d.spaces}")
# print(f"'a' in d.keys(): {'a' in d.keys()}") # keys() might not exist on older gym versions or might return something else
try:
    print(f"d.keys(): {d.keys()}")
    print(f"'a' in d.keys(): {'a' in d.keys()}")
except AttributeError:
    print("d.keys() does not exist")
