import hydra
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
import pickle
from PIL import Image
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import shutil
from joint_marker import JointMarker
from render_data import RenderData, RenderDataParallel

os.environ["PYOPENGL_PLATFORM"] = "egl"

JOINT_COLOR_MAP = {
    1: "red",
    3: "green",
    5: "purple",
}


class RenderDataNoTex(RenderData):
    """
    RenderData class to draw actions in images of a RLBench dataset.
    This version avoids loading textures if not needed.
    """

    def __init__(self, cfg) -> None:
        """
        Initializes an instance of RenderData class

        Parameters:
        - cfg (DictConfig): configuration for RenderData

        Returns:
        - None
        """
        self.cfg = cfg
        # Initialize texture files only if random background is enabled
        if self.cfg.draw.rnd_bg:
            textures_path = cfg.textures_path
            self._texture_files = [
                os.path.join(textures_path, f) for f in os.listdir(textures_path)
            ]
        self._image_width = cfg.image_width
        self._image_height = cfg.image_height

        # Create new directory to save if specfied
        # Otherwise, modify based on existing directory name
        dataset_root = self.cfg.dataset_root
        full_dataset_root = os.path.abspath(dataset_root)
        full_dataset_root = (
            full_dataset_root[:-1]
            if full_dataset_root[-1] == "/"
            else full_dataset_root
        )
        dataset_dir_name = full_dataset_root.split("/")[-1]
        self._full_context_dataset_dir_name = dataset_dir_name + "_rgb_rendered"
        self._random_context_dataset_dir_name = dataset_dir_name + "_rnd_bg"

        if self.cfg.save_path is not None:
            self._parent_path = self.cfg.save_path
        else:
            self._parent_path = os.path.dirname(full_dataset_root)

        # Copy data from the original directory
        if self.cfg.draw.rgb_rendered:
            self._full_context_dst_path = os.path.join(
                self._parent_path, self._full_context_dataset_dir_name
            )
            self.ensure_folder_exists(self._full_context_dst_path)
            self.deepcopy_folder(
                os.path.join(full_dataset_root, self.cfg.task),
                os.path.join(self._full_context_dst_path, self.cfg.task),
            )

        if self.cfg.draw.rnd_bg:
            self.ensure_folder_exists(
                os.path.join(self._parent_path, self._random_context_dataset_dir_name)
            )
            self._random_context_dst_path = os.path.join(
                self._parent_path, self._random_context_dataset_dir_name
            )
            self.deepcopy_folder(
                os.path.join(full_dataset_root, self.cfg.task),
                os.path.join(self._random_context_dst_path, self.cfg.task),
            )


class RenderDataNoTexParallel(RenderDataNoTex, RenderDataParallel):
    """
    Child class of RenderData to enable parallel generation
    """

    def __init__(self, cfg) -> None:
        """
        Initializes an instance of RenderDataParallel class

        Parameters:
        - cfg (DictConfig): configuration for RenderData

        Returns:
        - None
        """
        super().__init__(cfg)


@hydra.main(config_path="cfgs", config_name="render", version_base=None)
def render_data_no_tex(cfg):
    """
    Action rendering entrypoint
    """
    print(cfg)

    import time

    render_start = time.time()
    render_data = RenderDataNoTexParallel(cfg)
    render_data.generate()
    render_end = time.time()
    print(f"Render time: {render_end - render_start}")


if __name__ == "__main__":
    render_data_no_tex()
