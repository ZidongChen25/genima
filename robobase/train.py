from pathlib import Path

import hydra


def _find_resume_snapshot(work_dir: Path, current_dir: Path):
    candidates = []
    for base_dir in (work_dir, current_dir):
        candidates.append(base_dir / "snapshot.pt")
        candidates.append(base_dir / "snapshots" / "latest_snapshot.pt")

    seen = set()
    for snapshot in candidates:
        snapshot = snapshot.resolve()
        if snapshot in seen:
            continue
        seen.add(snapshot)
        if snapshot.exists():
            return snapshot
    return None


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.workspace import Workspace

    workspace = Workspace(cfg)
    snapshot = _find_resume_snapshot(workspace.work_dir, Path.cwd())
    if snapshot is not None:
        print(f"resuming: {snapshot}")
        workspace.load_snapshot(snapshot, preserve_runtime_cfg=True)
    workspace.train()


if __name__ == "__main__":
    main()
