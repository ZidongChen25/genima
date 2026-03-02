import argparse
import json
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an existing RoboBase snapshot.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a training run directory that contains .hydra/config.yaml.",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="latest_snapshot.pt",
        help=(
            "Snapshot file path or name. If relative and exists under "
            "<run-dir>/snapshots, it will be resolved there."
        ),
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=50,
        help="Total number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--num-parallel-envs",
        type=int,
        default=1,
        help="Number of parallel evaluator processes (same snapshot).",
    )
    parser.add_argument(
        "--execution-length",
        type=int,
        default=None,
        help="Optional override for execution_length during evaluation.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task override, e.g. take_lid_off_saucepan.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional dataset root override for RLBench env.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Enable eval video saving.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging for this evaluation run.",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="eval.config",
        help="Path to save compact eval summary config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode evaluation progress.",
    )
    return parser.parse_args()


def _resolve_snapshot_path(run_dir: Path, snapshot: str) -> Path:
    snapshot_path = Path(snapshot)
    if snapshot_path.is_file():
        return snapshot_path.resolve()

    if not snapshot_path.is_absolute():
        candidate = run_dir / "snapshots" / snapshot
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find snapshot '{snapshot}'. "
        f"Checked absolute path and '{run_dir / 'snapshots' / snapshot}'."
    )


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _apply_eval_overrides(cfg, args):
    with open_dict(cfg):
        # Keep workspace initialization path consistent with single-process eval.
        # Parallelism is handled by one independent process per evaluator.
        cfg.log_eval_video = bool(args.save_video)
        cfg.wandb.use = bool(args.use_wandb)
        cfg.tb.use = False
        cfg.eval_verbose = bool(args.verbose)
        if args.execution_length is not None:
            cfg.execution_length = int(args.execution_length)
        if args.task is not None:
            cfg.env.task_name = args.task
        if args.dataset_root is not None:
            cfg.env.dataset_root = args.dataset_root


def _split_episodes(total_episodes: int, workers: int):
    workers = max(1, int(workers))
    base = total_episodes // workers
    rem = total_episodes % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _eval_worker(payload):
    run_dir, snapshot_path, args_dict, num_eval_episodes, worker_id = payload
    args = argparse.Namespace(**args_dict)
    payload_obj = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    cfg = payload_obj.get("cfg", None)
    if cfg is None:
        cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
    _apply_eval_overrides(cfg, args)
    with open_dict(cfg):
        cfg.num_eval_episodes = int(num_eval_episodes)
        cfg.seed = int(cfg.seed) + int(worker_id)
    from robobase.workspace import Workspace

    workspace = Workspace(cfg, work_dir=str(run_dir))
    try:
        workspace.load_snapshot(str(snapshot_path))
        _apply_eval_overrides(workspace.cfg, args)
        with open_dict(workspace.cfg):
            workspace.cfg.num_eval_episodes = int(num_eval_episodes)
            workspace.cfg.seed = int(workspace.cfg.seed) + int(worker_id)
        metrics = workspace.eval()
    finally:
        workspace.shutdown()
    metrics = _to_jsonable(metrics)
    return metrics


def _aggregate_metrics(worker_metrics, worker_episodes):
    total_eps = int(sum(worker_episodes))
    result = {
        "num_parallel_envs": len(worker_metrics),
        "num_eval_episodes": total_eps,
        "worker_episodes": worker_episodes,
        "worker_metrics": worker_metrics,
    }
    for key in ("episode_success", "episode_reward", "episode_length"):
        vals = [m.get(key, None) for m in worker_metrics]
        if any(v is None for v in vals):
            continue
        result[key] = float(
            sum(float(v) * int(n) for v, n in zip(vals, worker_episodes)) / max(total_eps, 1)
        )
    return result


def _eval_worker_entry(payload, out_queue):
    worker_id = int(payload[4])
    try:
        metrics = _eval_worker(payload)
        out_queue.put((worker_id, metrics, None))
    except Exception:
        out_queue.put((worker_id, None, traceback.format_exc()))


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Cannot find Hydra config at '{cfg_path}'.")

    snapshot_path = _resolve_snapshot_path(run_dir, args.snapshot)
    if args.num_parallel_envs < 1:
        raise ValueError("--num-parallel-envs must be >= 1.")
    if args.num_eval_episodes < 1:
        raise ValueError("--num-eval-episodes must be >= 1.")
    if args.num_parallel_envs > 1 and args.use_wandb:
        raise ValueError("Parallel eval does not support --use-wandb.")
    if args.num_parallel_envs > 1 and args.save_video:
        raise ValueError("Parallel eval does not support --save-video.")

    if args.num_parallel_envs == 1:
        worker_metrics = [
            _eval_worker(
                (
                    str(run_dir),
                    str(snapshot_path),
                    vars(args),
                    int(args.num_eval_episodes),
                    0,
                )
            )
        ]
        metrics = _aggregate_metrics(worker_metrics, [int(args.num_eval_episodes)])
    else:
        episodes_per_worker = _split_episodes(
            int(args.num_eval_episodes), int(args.num_parallel_envs)
        )
        jobs = []
        for worker_id, n_episodes in enumerate(episodes_per_worker):
            if n_episodes <= 0:
                continue
            jobs.append(
                (
                    str(run_dir),
                    str(snapshot_path),
                    vars(args),
                    int(n_episodes),
                    int(worker_id),
                )
            )
        ctx = mp.get_context("spawn")
        out_queue = ctx.Queue()
        procs = [
            ctx.Process(target=_eval_worker_entry, args=(job, out_queue))
            for job in jobs
        ]

        for p in procs:
            p.start()

        results = []
        errors = []
        for _ in procs:
            worker_id, worker_metrics, err = out_queue.get()
            if err is None:
                results.append((worker_id, worker_metrics))
            else:
                errors.append((worker_id, err))

        for p in procs:
            p.join()

        for p in procs:
            if p.exitcode not in (0, None):
                errors.append((None, f"Worker exited with code {p.exitcode}"))

        if errors:
            error_msg = "\n\n".join(
                [f"[worker {wid}] {msg}" for wid, msg in errors]
            )
            raise RuntimeError(f"Parallel evaluation failed:\n{error_msg}")

        results.sort(key=lambda x: x[0])
        worker_metrics = [m for _, m in results]
        metrics = _aggregate_metrics(worker_metrics, [j[3] for j in jobs])

    success_rate = metrics.get("episode_success", None)
    if success_rate is not None:
        print(f"success_rate={success_rate:.6f}")
    else:
        print("success_rate=None")

    out_path = Path(args.output_config).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        "run_dir": str(run_dir),
        "snapshot": str(snapshot_path),
        "execution_length": (
            int(args.execution_length) if args.execution_length is not None else None
        ),
        "num_eval_episodes": int(args.num_eval_episodes),
        "num_parallel_envs": int(args.num_parallel_envs),
        "success_rate": success_rate,
        "worker_episodes": metrics.get("worker_episodes", []),
        "worker_episode_success": [
            m.get("episode_success", None) for m in metrics.get("worker_metrics", [])
        ],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, sort_keys=True)
    print(f"saved_eval_config={out_path}")


if __name__ == "__main__":
    main()
