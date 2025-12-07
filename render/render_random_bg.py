import hydra
from render_data import RenderDataParallel
import time

@hydra.main(config_path="cfgs", config_name="render", version_base=None)
def main(cfg):
    # Force configuration for random background rendering
    cfg.draw.rgb_rendered = False
    cfg.draw.rnd_bg = True
    
    print("Starting Random Background Rendering...")
    print(f"Dataset Root: {cfg.dataset_root}")
    print(f"Textures Path: {cfg.textures_path}")
    print(f"Task: {cfg.task}")
    
    render_start = time.time()
    render_data = RenderDataParallel(cfg)
    render_data.generate()
    render_end = time.time()
    print(f"Render time: {render_end - render_start}")

if __name__ == "__main__":
    main()
