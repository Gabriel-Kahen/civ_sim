# civ_sim

`civ_sim` is a persistent, sprite-based 2D world where evolving agents survive by extracting, transporting, storing, processing, and maintaining resources. The target artifact is visible proto-urban structure: roads, depots, workshops, homes, chokepoints, districts, and shifting settlement frontiers.

## Current Scope

The first implementation includes:

- deterministic patch-seeded chunk generation on an infinite plane
- dynamic chunk activation through settlement influence
- a discrete tile simulation with persistent structures and inventories
- evolving recurrent controllers with scalar trait genes
- asynchronous birth/death, mutation, and local ecological selection
- interactive `pygame-ce` rendering with overlays
- headless experiment mode, save/load, and frame/map export

## Quick Start

Create a virtual environment, install the package, then run either the sandbox or a headless experiment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
civ-sim sandbox --ticks 5000
civ-sim experiment --ticks 20000 --export-every 50 --output exports/run1
```

The default balance file is `configs/default.yaml`. Use `--config path/to/file.yaml`
to run alternate survival/building/economy tuning without editing code.
Video exports use the fixed viewport controls in that YAML file
(`video_center_x`, `video_center_y`, `video_view_width_tiles`,
`video_view_height_tiles`, and `video_pixels_per_tile`) so map expansion does
not rescale or recenter the output mid-run.
The same YAML also controls performance-sensitive sampling intervals:
`resource_update_interval`, `traffic_decay_interval`, `influence_update_interval`,
and `detailed_metrics_interval`. Slow processes are scaled by their interval
where applicable.

Experiment checkpoints default to binary `.pkl` files because full genomes make
text JSON snapshots very large. Use `.pkl.gz` for smaller compressed saves, or
`.json` when a human-readable legacy snapshot is needed.

For long experiments, sample video and checkpoint periodically instead of
rendering every tick:

```bash
civ-sim experiment \
  --ticks 100000 \
  --video-every 25 \
  --checkpoint-every 10000 \
  --export-every 1000 \
  --log-every 500 \
  --output exports/longrun
```

For maximum simulation throughput, decouple the run from rendering:

```bash
civ-sim experiment \
  --ticks 100000 \
  --checkpoint-every 10000 \
  --metrics-every 25 \
  --log-every 500 \
  --no-video \
  --no-frames \
  --no-maps \
  --no-analysis \
  --output exports/longrun_headless

civ-sim render-checkpoints \
  --input exports/longrun_headless/checkpoints \
  --output exports/longrun_headless/checkpoint_video.mp4
```

Add `--no-save` only when periodic checkpoints are enough and you do not need
an extra final-state pickle.

Experiment runs write `run_manifest.json`, `config.yaml`, `run.log`, streaming
`metrics.csv`, sampled video, periodic checkpoints, final maps, and
`final_state.pkl`.

## Discord Extinction Alerts

Headless experiment mode can send a Discord message if the population drops to
zero. Create a Discord webhook in the target channel, then set the webhook URL
in the process environment before launching the run:

```bash
export CIV_SIM_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
civ-sim experiment --ticks 200000 --output exports/longrun
```

Do not commit the webhook URL into `configs/default.yaml` or the repo. If the
environment variable is missing, the run continues normally and records
`discord_webhook_missing` in `run.log` when extinction happens.

## Controls

In sandbox mode:

- `space`: pause or resume
- `tab`: cycle overlays
- `s`: save the world state
- `e`: export the current derived maps
- `r`: reset to a fresh world
- `esc`: quit

## Architecture

- `configs/default.yaml`: primary tuning file for simulation, survival, economy, and structure balance
- `config.py`: typed config model and YAML loading helpers
- `constants.py`: enums and static lookup tables
- `models.py`: state containers and serialization helpers
- `worldgen.py`: deterministic terrain/resource generation
- `world.py`: chunk management, influence, and terrain access
- `controller.py`: recurrent controller execution and genome mutation
- `sim.py`: main ecological simulation loop
- `render.py`: sprite generation, overlays, and frame export
- `analysis.py`: metrics, derived maps, and export helpers
- `io.py`: save/load helpers
- `cli.py`: command-line entrypoints
