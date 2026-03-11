Use `xvfb-run` for headless RLBench/CoppeliaSim training.
Do not set `QT_QPA_PLATFORM=offscreen` (it cannot create the OpenGL context needed here).

```bash
unset QT_QPA_PLATFORM
  xvfb-run -a -s "-screen 0 1024x768x24" env \
    COPPELIASIM_ROOT="$HOME/.local/bin/CoppeliaSim" \
    LD_LIBRARY_PATH="$HOME/.local/bin/CoppeliaSim:${LD_LIBRARY_PATH}" \
    QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/.local/bin/CoppeliaSim" \
    python train.py \
    hydra/launcher=basic \
    experiment_name=open_grill_h16 \
    'hydra.run.dir=./exp_local/${now:%Y.%m.%d}/${now:%H%M%S}_${experiment_name}' \
    'hydra.sweep.dir=./exp_local/${now:%Y.%m.%d}/${now:%H%M}_${experiment_name}' \
    method=diffusion \
    env=rlbench/open_grill \
    env.dataset_root=../data/train_data \
    env.action_mode=JOINT_POSITION \
    env.renderer=opengl \
    demos=50 \
    pixels=true \
    is_imitation_learning=true \
    num_pretrain_steps=100000 \
    num_train_frames=0 \
    action_sequence=16 \
    execution_length=8 \
    replay.nstep=1 \
    batch_size=64 \
    eval_every_steps=10000 \
    num_eval_episodes=50 \
    save_snapshot=true \
    snapshot_every_n=10000 \
    use_min_max_normalization=true \
    num_gpus=1
```
unset QT_QPA_PLATFORM
  xvfb-run -a -s "-screen 0 1024x768x24" env \
    COPPELIASIM_ROOT="$HOME/.local/bin/CoppeliaSim" \
    LD_LIBRARY_PATH="$HOME/.local/bin/CoppeliaSim:${LD_LIBRARY_PATH}" \
    QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/.local/bin/CoppeliaSim" \
    python train.py \
    hydra/launcher=basic \
    experiment_name=lid_off_diff50_noise \
    'hydra.run.dir=./exp_local/${now:%Y.%m.%d}/${now:%H%M%S}_${experiment_name}' \
    'hydra.sweep.dir=./exp_local/${now:%Y.%m.%d}/${now:%H%M}_${experiment_name}' \
    method=diffusion \
    env=rlbench/take_lid_off_saucepan \
    env.dataset_root=../data/train_data \
    env.action_mode=JOINT_POSITION \
    env.renderer=opengl \
    demos=50 \
    pixels=true \
    is_imitation_learning=true \
    num_pretrain_steps=100000 \
    num_train_frames=0 \
    action_sequence=16 \
    execution_length=8 \
    replay.nstep=1 \
    batch_size=64 \
    eval_every_steps=10000 \
    num_eval_episodes=50 \
    save_snapshot=true \
    snapshot_every_n=10000 \
    use_min_max_normalization=true \
    replay.tail_noise_n=8 \
    num_gpus=1



xvfb-run -a -s "-screen 0 1024x768x24" env \
    COPPELIASIM_ROOT="$HOME/.local/bin/CoppeliaSim" \
    LD_LIBRARY_PATH="$HOME/.local/bin/CoppeliaSim:${LD_LIBRARY_PATH}" \
    QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/.local/bin/CoppeliaSim" \
    python eval_snapshot.py \
      --snapshot /home/zc1525/Desktop/genima/robobase/exp_local/2026.03.01/235706_lamp_on_diff50_a16/snapshots/10000_snapshot.pt \
      --num-eval-episodes 50 \
      --num-parallel-envs 10 \
      --num-eval-envs 1 \
      --verbose

Flow matching training command for the same setup:

```bash
unset QT_QPA_PLATFORM
  xvfb-run -a -s "-screen 0 1024x768x24" env \
    COPPELIASIM_ROOT="$HOME/.local/bin/CoppeliaSim" \
    LD_LIBRARY_PATH="$HOME/.local/bin/CoppeliaSim:${LD_LIBRARY_PATH}" \
    QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/.local/bin/CoppeliaSim" \
    python train.py \
    hydra/launcher=basic \
    experiment_name=open_grill_h16_flow_matching \
    'hydra.run.dir=./exp_local/${now:%Y.%m.%d}/${now:%H%M%S}_${experiment_name}' \
    'hydra.sweep.dir=./exp_local/${now:%Y.%m.%d}/${now:%H%M}_${experiment_name}' \
    method=flow_matching \
    env=rlbench/open_grill \
    env.dataset_root=../data/train_data \
    env.action_mode=JOINT_POSITION \
    env.renderer=opengl \
    demos=50 \
    pixels=true \
    is_imitation_learning=true \
    num_pretrain_steps=100000 \
    num_train_frames=0 \
    action_sequence=16 \
    execution_length=8 \
    replay.nstep=1 \
    batch_size=64 \
    eval_every_steps=10000 \
    num_eval_episodes=50 \
    save_snapshot=true \
    snapshot_every_n=10000 \
    use_min_max_normalization=true \
    num_gpus=1
```
RESUME=1 NUM_PRETRAIN_STEPS=200000 bash robobase/run_rlbench_diffusion_sweep.sh