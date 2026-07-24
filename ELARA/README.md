# ELARA: Independent Simulation and PPO Implementation

This directory is a clean implementation of the revised serving-satellite
selection design.  It does not import code or scripts from `Simulation/`.

## Implemented design

- Every request gathers all satellites hosting replicas of every requested
  microservice.
- Source, destination, replica hosts, and the necessary connector relays form a
  deduplicated request graph.
- A rooted shortest-path connector tree guarantees that all terminal nodes are
  connected. Relay nodes are context only and never become serving actions.
- Connector maintenance is lazy: construct on request arrival or placement
  change, refresh measurements at ordinary stages, and repair only after a slot
  transition disconnects the current tree.
- Current and future connectivity use sparse `edge_index` tensors. Future
  observations contain predictable orbital connectivity only.
- The shared candidate scorer receives normalized hop distance, bottleneck
  rate, and candidate computing queue in addition to learned graph/request
  embeddings.
- After serving selection, a capacity-aware min-cost splittable-flow router
  reserves complete paths within each slot. Undelivered data remains at the
  stage source and is rerouted under the next topology snapshot.
- Initial replica counts are independently sampled per service and constrained
  by satellite memory. At every adaptation window, service pressure ranks
  costly services and a shared contextual-UCB policy selects one fully
  specified `no_op`, `relocate`, `scale_out`, or `scale_in` action. Scale
  actions change the count by exactly one; relocation preserves it.
- A fixed service-chain catalog (lengths 5, 10, and 15 by default) feeds a
  chronological Poisson arrival process. Compute and ISL background load evolve
  through slot-correlated Markov states, and reservations persist across
  overlapping requests.
- The last action includes final-output routing to the destination in its
  latency, energy, and reward.

## Layout

- `config.py`: all experiment/model parameters.
- `domain.py`: service, request, and resource models.
- `topology.py`: CSV trace loader and sparse temporal graph.
- `connector.py`: connected request subgraph and lazy repair.
- `routing.py`: cross-slot min-cost splittable flow and capacity reservation.
- `background.py`: Markov compute and ISL background-load processes.
- `bandit.py`: trace pressure, target-plane resolution, four-action LinUCB, and
  dynamic replica-number/placement adaptation.
- `state.py`: variable-size sparse PPO observation.
- `environment.py`: request execution, routing, computation, reward, and update.
- `model.py`: request encoder, edge-aware spatial graph attention, temporal
  attention, request-conditioned global pooling, actor, and critic.
- `ppo.py`: PPO rollout, GAE, clipped objective, checkpointing.
- `train.py`, `evaluate.py`: command-line entry points.
- `scripts/`: reproducible shell entry points.
- `tests/`: standard-library unit and integration tests.

## Environment

Python 3.10+ is recommended.

```bash
python3 -m venv .venv-elara
source .venv-elara/bin/activate
python3 -m pip install -r ELARA/requirements.txt
```

The default topology trace is read from:

```text
WalkerDeltaConstellationSimu/Walker_Delta_ISL_Simu.csv
```

The path can be changed through `ELARAConfig.trace_csv`.

## Tests

Tests do not require PyTorch:

```bash
ELARA/scripts/test.sh
```

## Greedy/random smoke evaluation

```bash
ELARA/scripts/evaluate.sh \
  --policy greedy \
  --episodes 10 \
  --max-trace-slots 20 \
  --output-dir ELARA/outputs/greedy-smoke
```

Use `--policy random` for the random serving baseline.

## PPO training

```bash
ELARA/scripts/train.sh \
  --max-trace-slots 606 \
  --request-template-lengths 5,10,15 \
  --arrival-lambda 0.35 \
  --replica-min 5 \
  --replica-max 10 \
  --future-horizon 3 \
  --ppo-minibatch-size 16 \
  --ppo-update-interval-slots 5 \
  --ppo-transaction-history-slots 10 \
  --ppo-transaction-max-reuse 2 \
  --pretrain-cycles 1 \
  --joint-training-cycles 1 \
  --background-load-scale 0.5 \
  --route-horizon 3 \
  --route-max-paths 3 \
  --adaptation-window-slots 10 \
  --adaptation-top-k 10 \
  --output-dir ELARA/outputs/train-seed42
```

Outputs include `config.json`, `training_metrics.csv`,
`ppo_update_metrics.csv`, periodic `ppo_latest.pt`, the PPO-only
`ppo_pretrained.pt`, and final joint-training `ppo_final.pt`. Checkpoints
contain the PPO model, optimizer, shared LinUCB state, partially collected
deployment window, and the current replica placement.
`orchestration_summary.json` records both phase counts, PPO update count,
final bandit statistics, and deployment.

PPO keeps complete request trajectories from the latest ten time slots by
default. A transition can participate in at most two adjacent PPO updates.
This permits the update at a later time slot to reuse recent data while
bounding policy staleness and replay memory. The training loop restores
request-contiguous trajectory order before computing GAE.

Default training covers two consecutive uses of the loaded constellation
cycle. With the default trace, each cycle contains 606 time slots. Replica
adaptation is disabled during the first cycle so that PPO receives a stable
pretraining environment. PPO then continues training for a second cycle with
replica adaptation enabled. The Poisson arrival process determines how many
requests arrive in each phase, so training has no `--episodes` argument.
`training_metrics.csv` identifies the phase and contains one row per admitted
request.

PPO is updated after every five time slots that contain collected transitions.
Updates use logical mini batches. Variable sized sparse request graphs are
encoded independently inside each mini batch, then their losses are averaged
before one backward pass and optimizer step. The default mini batch size is 16.
During an update, the parallel launcher reports `PPO updating` in the progress
line and returns to request processing when the update finishes.

### Plot PPO reward and loss curves

The plotting utility accepts one `training_metrics.csv`, a single task
directory, or a parallel training root containing multiple tasks. For multiple
seeds, it plots the mean and 95% confidence interval. Reward points are aligned
by episode, while sparse loss records are aligned by PPO update index.

```bash
ELARA/scripts/plot_ppo_curves.sh \
  ELARA/outputs/parallel-train/20260723-081334 \
  --reward-window 25 \
  --loss-window 1 \
  --show-runs
```

On Windows, use `ELARA\scripts\plot_ppo_curves.ps1` with the same arguments.
The script generates combined reward and total-loss curves, a standalone reward
curve, and policy, value, total-loss, and entropy panels in both PNG and PDF.

## PPO evaluation

```bash
ELARA/scripts/evaluate.sh \
  --policy ppo \
  --checkpoint ELARA/outputs/train-seed42/ppo_final.pt \
  --seed 42 \
  --episodes 100 \
  --output-dir ELARA/outputs/test-seed42
```

## All-baseline testing

The cross-platform baseline runner evaluates `ELARA`, `ELARA-NB`, `ELARA-NR`,
`ELARA-SH`, `SECO`, `SP-Routing`, and `SC-NFV` through the common Simulation
evaluation environment. Each baseline and seed pair is an independent task.
The default concurrency is four, and CUDA tasks are assigned round robin over
all visible GPUs. CPU and MPS are also supported. All metrics, logs, manifests,
and merged CSV files are written below `ELARA`.

```bash
ELARA/scripts/test_baselines.sh \
  --device auto \
  --tasks 4 \
  --seeds 42,43,44,45 \
  --model-root Simulation/multi_seed_runs
```

On Windows:

```powershell
ELARA\scripts\test_baselines.ps1 `
  --device auto `
  --tasks 4 `
  --seeds 42,43,44,45 `
  --model-root Simulation\multi_seed_runs
```

Use `--max-slots 5` for a short smoke test. The default output directory is
`ELARA/outputs/baseline-tests/<timestamp>`. PPO checkpoints are required for
`ELARA`, `ELARA-NB`, and `ELARA-SH`; an intentionally untrained smoke test must
explicitly pass `--no-load-checkpoint`.

## Baseline contribution plots

The contribution plotter automatically selects the latest complete baseline
run, aggregates the random seeds with 95% confidence intervals, and writes
ablation and comparison figures into separate directories. It covers overall
latency and energy, relative improvement, routing and reliability, service
chain length, temporal behavior across all 606 slots, request tail
distributions, and communication/computation cost decomposition.

```bash
ELARA/scripts/plot_baseline_contributions.sh \
  ELARA/outputs/baseline-tests \
  --rolling-window 25
```

On Windows:

```powershell
ELARA\scripts\plot_baseline_contributions.ps1 `
  ELARA\outputs\baseline-tests `
  --rolling-window 25
```

By default, both PNG and PDF figures are produced below
`<run>/contribution_plots/ablation` and
`<run>/contribution_plots/comparison`. The parent directory also receives
machine-readable CSV and JSON summaries and a short Markdown interpretation.
Use `--formats png,pdf,svg` to change the output formats.

## Parameter sensitivity experiments

All parameter sensitivity runs use the fixed catalog at
`ELARA/data/request_templates_seed2026.json`. The catalog contains fourteen
templates with chain lengths 5, 10, and 15 in an 8:4:2 ratio. The Poisson
process controls how often a catalog entry is sampled. Communication volumes
are 20--200 MB with an approximately 80 MB mean. Sensitivity training uses one
606-slot PPO-only cycle followed by one 606-slot joint-training cycle.
Full-cycle testing derives the request count from all arrivals within one
606-slot cycle.

Regenerate the catalog, if required, with a different catalog seed:

```bash
ELARA/scripts/generate_request_templates.sh \
  --output ELARA/data/request_templates_seed2026.json \
  --seed 2026
```

Before final testing, tune only on validation seeds that are not used in the
reported experiments:

```bash
ELARA/scripts/tune_hyperparameters.sh \
  --validation-seeds 202,203 \
  --tasks 2 \
  --device auto \
  --max-trace-slots 606 \
  --output-root ELARA/outputs/tuning/validation
```

The optional tuner compares the auditable profiles in
`ELARA/configs/hyperparameter_search.json` and writes
`tuning_results.csv`. It does not generate or consume a frozen profile file.
Final sensitivity experiments use the explicit scenario parameters supplied
to the sensitivity runner.

Run all final sensitivity training and testing:

```bash
ELARA/scripts/run_sensitivity.sh \
  --seeds 42,43,44,45 \
  --weights 0.5:0.5,0.35:0.65,0.65:0.35 \
  --route-max-paths 3,5,7 \
  --train-tasks 2 \
  --test-tasks 4 \
  --device auto \
  --max-trace-slots 606 \
  --output-root ELARA/outputs/sensitivity/final
```

The weight experiment fixes the maximum augmenting-path count at three. The
routing experiment fixes the latency and energy weights at 0.5:0.5. Therefore
the two categories each contain three conditions and four seeds, rather than
an unnecessary nine-condition Cartesian product.

Training and testing can also be launched separately by passing the same
output root:

```bash
ELARA/scripts/train_sensitivity.sh \
  --train-tasks 2 \
  --output-root ELARA/outputs/sensitivity/final

ELARA/scripts/test_sensitivity.sh \
  --test-tasks 4 \
  --output-root ELARA/outputs/sensitivity/final
```

After testing, draw the two experiment categories in separate directories:

```bash
ELARA/scripts/plot_sensitivity.sh \
  ELARA/outputs/sensitivity/final
```

The output includes latency, energy, success rate, the latency-energy
tradeoff, actual augmenting-path use, and cross-slot routing. PNG and PDF are
generated by default. Equivalent PowerShell scripts with `.ps1` suffixes are
provided for Windows.

## Parallel training and evaluation

The parallel launchers detect visible CUDA GPUs and Apple Metal Performance
Shaders (MPS) automatically. `--device auto` prefers CUDA, then MPS, then CPU.
CUDA jobs are assigned round-robin, so when the task count exceeds the GPU
count, each GPU receives nearly the same number of processes. MPS is treated as
one shared accelerator, and with no accelerator all jobs run on CPU. Parallel
training defaults to two tasks, while parallel evaluation defaults to four.
Explicit `--device cuda`, `--device mps`, and `--device cpu` selections are
also supported.

```bash
# Apple Silicon: two independent jobs share the MPS accelerator.
ELARA/scripts/train_parallel.sh --device mps --tasks 2 --max-trace-slots 606

# Single-process training and PPO evaluation can select MPS directly.
ELARA/scripts/train.sh --device mps --max-trace-slots 606
ELARA/scripts/evaluate.sh --device mps --policy ppo --checkpoint MODEL.pt
```

```bash
# Two concurrent training jobs with seeds 42 and 43.
ELARA/scripts/train_parallel.sh \
  --tasks 2 \
  --base-seed 42 \
  --output-root ELARA/outputs/train-batch \
  --max-trace-slots 606

# Four concurrent evaluations of one checkpoint with different request seeds.
ELARA/scripts/evaluate_parallel.sh \
  --base-seed 1000 \
  --output-root ELARA/outputs/test-batch \
  --policy ppo \
  --checkpoint ELARA/outputs/train-seed42/ppo_final.pt \
  --episodes 100
```

Each task has an independent output directory and log file. The output root
also contains `launcher_manifest.json`, which records commands, seeds, GPU
assignments, and exit status. Use `--dry-run` to inspect assignments without
starting jobs. `CUDA_VISIBLE_DEVICES` is honored when it is already set. While
jobs run, the launcher displays an aggregate episode progress bar, completed
task count, elapsed time, and estimated remaining time. ETA is computed from
the observed speed of each active job and uses the slowest remaining job as the
parallel batch estimate.

### Windows Command Prompt

Windows users can run the `.cmd` launchers from Command Prompt or Windows
Terminal. They use the active environment's `python` first and fall back to
`py -3` automatically. The scripts can be launched from any working directory.

```bat
REM Two parallel CUDA training jobs. Use --device auto for CUDA/CPU fallback.
ELARA\scripts\train_parallel.cmd ^
  --device auto ^
  --tasks 2 ^
  --base-seed 42 ^
  --output-root ELARA\outputs\train-batch ^
  --max-trace-slots 606

REM Parallel PPO evaluation.
ELARA\scripts\evaluate_parallel.cmd ^
  --device auto ^
  --tasks 4 ^
  --base-seed 1000 ^
  --output-root ELARA\outputs\test-batch ^
  --policy ppo ^
  --checkpoint ELARA\outputs\train-seed42\ppo_final.pt ^
  --episodes 100
```

MPS is an Apple platform feature and is not available on Windows. On Windows,
`--device auto` selects CUDA when available and otherwise uses CPU. GPU
round-robin assignment, progress reporting, ETA, per-task logs, and the
launcher manifest behave the same as on Linux and macOS.

## Joint encoding

For each state, the model:

1. embeds the masked service sequence and normalized request features;
2. performs edge-aware graph attention over current sparse ISLs;
3. applies the same spatial encoder to predictable future sparse topologies;
4. fuses per-node snapshot embeddings using temporal attention;
5. computes request-conditioned global graph representation `h_g` once;
6. scores every valid service replica with one shared scorer;
7. applies the candidate action mask and softmax;
8. estimates `V(s)` with a separate critic head.

The graph computation is linear in sparse edge count. Since each satellite has
at most four ISLs, `E = O(N)` for a request graph.

## Integrated control loop

For every service stage:

1. PPO selects one valid replica from the connected request graph.
2. The data-routing module reserves one or more min-cost paths. A committed
   block must reach the serving satellite within its active slot.
3. The selected satellite executes the service and updates compute state.
4. Real routing, queueing, computation, latency, and energy measurements are
   added to the current deployment window.
5. The PPO state is refreshed at the actual completion time.

At a deployment-window boundary:

1. services are ranked by impact and route/queue/imbalance pressure;
2. cumulative stage-cost contribution selects the highest-impact replica for
   relocation and the lowest-impact replica for scale-in;
3. each orbital plane contributes at most one feasible low-load target, and
   historical traces estimate the best relocate/scale-out realization;
4. up to `adaptation_top_k_services` receive one four-action LinUCB decision;
5. scale-out/in adds/removes exactly one replica; relocation replaces one;
6. the next request rebuilds its terminals, relay connector, candidates, and
   action mask from the new deployment;
7. the following window updates LinUCB from normalized cost improvement.

Pass `--disable-adaptation` to training or evaluation to keep placement fixed.
