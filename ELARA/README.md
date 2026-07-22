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
- Three fixed service-chain templates (lengths 5, 10, and 15 by default) feed a
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
  --route-horizon 3 \
  --route-max-paths 3 \
  --adaptation-window-slots 10 \
  --adaptation-top-k 10 \
  --output-dir ELARA/outputs/train-seed42
```

Outputs include `config.json`, `training_metrics.csv`, periodic
`ppo_latest.pt`, and final `ppo_final.pt`. Checkpoints contain the PPO model,
optimizer, shared LinUCB state, partially collected deployment window, and the
current replica placement. `orchestration_summary.json` records final bandit
statistics and deployment.

Training always covers one loaded constellation cycle. With the default trace,
this cycle contains 606 time slots. The Poisson arrival process determines how
many requests arrive before the cycle boundary, so training has no
`--episodes` argument. `training_metrics.csv` contains one row for every
admitted request in that cycle.

## PPO evaluation

```bash
ELARA/scripts/evaluate.sh \
  --policy ppo \
  --checkpoint ELARA/outputs/train-seed42/ppo_final.pt \
  --seed 42 \
  --episodes 100 \
  --output-dir ELARA/outputs/test-seed42
```

## Parallel training and evaluation

The parallel launchers detect visible CUDA GPUs and Apple Metal Performance
Shaders (MPS) automatically. `--device auto` prefers CUDA, then MPS, then CPU.
CUDA jobs are assigned round-robin, so when the task count exceeds the GPU
count, each GPU receives nearly the same number of processes. MPS is treated as
one shared accelerator, and with no accelerator all jobs run on CPU. The
default task count is four. Explicit `--device cuda`, `--device mps`, and
`--device cpu` selections are also supported.

```bash
# Apple Silicon: four independent jobs share the MPS accelerator.
ELARA/scripts/train_parallel.sh --device mps --tasks 4 --max-trace-slots 606

# Single-process training and PPO evaluation can select MPS directly.
ELARA/scripts/train.sh --device mps --max-trace-slots 606
ELARA/scripts/evaluate.sh --device mps --policy ppo --checkpoint MODEL.pt
```

```bash
# Four concurrent training jobs with seeds 42, 43, 44, and 45.
ELARA/scripts/train_parallel.sh \
  --tasks 4 \
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
REM Four parallel CUDA training jobs. Use --device auto for CUDA/CPU fallback.
ELARA\scripts\train_parallel.cmd ^
  --device auto ^
  --tasks 4 ^
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
