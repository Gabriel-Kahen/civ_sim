from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import yaml
import torch

from civ_sim.config import SimConfig
from civ_sim.io import load_simulation, save_simulation
from civ_sim.sim import Simulation


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="civ-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sandbox = subparsers.add_parser("sandbox")
    _add_common_args(sandbox)
    sandbox.add_argument("--steps-per-frame", type=int, default=2)

    experiment = subparsers.add_parser("experiment")
    _add_common_args(experiment)
    experiment.add_argument("--output", type=Path, default=Path("exports/run"))
    experiment.add_argument("--save", type=Path)
    experiment.add_argument("--video-every", type=int, default=1, help="Render one video frame every N simulation ticks.")
    experiment.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint_NNNNNN.pkl every N ticks; 0 disables periodic checkpoints.")
    experiment.add_argument("--metrics-every", type=int, default=1, help="Append one metrics row every N ticks.")
    experiment.add_argument("--log-every", type=int, default=100, help="Append a human-readable progress log every N ticks.")
    experiment.add_argument("--no-video", action="store_true", help="Run headless without writing animation.mp4.")
    experiment.add_argument("--no-frames", action="store_true", help="Skip sampled PNG frame exports during the run.")
    experiment.add_argument("--no-maps", action="store_true", help="Skip derived map PNG exports at the end of the run.")
    experiment.add_argument("--no-analysis", action="store_true", help="Skip final JSON analysis exports.")
    experiment.add_argument("--no-save", action="store_true", help="Skip final_state.pkl; periodic checkpoints still work.")
    experiment.add_argument(
        "--stop-on-extinction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the experiment as soon as a nonzero population drops to zero.",
    )

    render_checkpoints = subparsers.add_parser("render-checkpoints")
    render_checkpoints.add_argument("--input", type=Path, required=True, help="Directory containing checkpoint_*.pkl files.")
    render_checkpoints.add_argument("--output", type=Path, required=True, help="Output mp4 path.")
    render_checkpoints.add_argument("--fps", type=int, default=20)
    render_checkpoints.add_argument("--limit", type=int, default=0, help="Maximum checkpoints to render; 0 renders all.")

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ticks", type=int, default=5000)
    parser.add_argument("--export-every", type=int, default=50)
    parser.add_argument("--load", type=Path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "sandbox":
        run_sandbox(args)
    elif args.command == "experiment":
        run_experiment(args)
    elif args.command == "render-checkpoints":
        run_render_checkpoints(args)


def _load_or_create(args) -> Simulation:
    if args.load:
        return load_simulation(args.load)
    config = SimConfig.from_yaml(args.config) if args.config and args.config.exists() else SimConfig()
    config.seed = args.seed
    config.export_every = args.export_every
    return Simulation.create(config)


def run_sandbox(args) -> None:
    import pygame
    from civ_sim.render import Renderer

    simulation = _load_or_create(args)
    renderer = Renderer(simulation.config)
    clock = pygame.time.Clock()
    paused = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_TAB:
                    renderer.cycle_overlay()
                elif event.key == pygame.K_s:
                    save_simulation(simulation, simulation.config.autosave_path)
                elif event.key == pygame.K_e:
                    output_dir = simulation.config.output_root / "manual_export"
                    renderer.export_maps(simulation, output_dir)
                    export_metrics(simulation, output_dir)
                elif event.key == pygame.K_r:
                    simulation.reset()

        if not paused:
            for _ in range(args.steps_per_frame):
                if args.ticks and simulation.current_tick >= args.ticks:
                    paused = True
                    break
                simulation.step()
        renderer.draw(simulation)
        clock.tick(simulation.config.fps)

    renderer.shutdown()


def run_experiment(args) -> None:
    pygame = None
    torch.set_grad_enabled(False)
    simulation = _load_or_create(args)
    needs_renderer = not (args.no_video and args.no_frames and args.no_maps)
    if needs_renderer:
        from civ_sim.render import Renderer

    renderer = Renderer(simulation.config, headless=True) if needs_renderer else None
    output_dir = args.output
    frames_dir = output_dir / "frames"
    maps_dir = output_dir / "maps"
    checkpoints_dir = output_dir / "checkpoints"
    video_path = output_dir / "animation.mp4"
    save_path = args.save or output_dir / "final_state.pkl"
    frames_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(args, simulation.config, output_dir)
    metrics_logger = MetricsLogger(output_dir / "metrics.csv", every=max(1, args.metrics_every))
    progress_logger = ProgressLogger(output_dir / "run.log", every=max(1, args.log_every))
    extinction_notifier = DiscordExtinctionNotifier.from_environment(output_dir)
    previous_population = len(simulation.agents)
    video_every = max(1, args.video_every)
    video_writer = None if args.no_video else renderer.open_video_writer(video_path, fps=20)
    started_at = time.perf_counter()

    try:
        while simulation.current_tick < args.ticks:
            stats = simulation.step()
            metrics_logger.write(simulation.current_tick, stats.to_dict())
            progress_logger.write(simulation, started_at)
            if previous_population > 0 and stats.population == 0:
                extinction_notifier.notify(simulation, stats, started_at)
            if args.stop_on_extinction and stats.population == 0:
                progress_logger.write_extinction_stop(simulation, started_at)
                break
            previous_population = stats.population
            if len(simulation.stats_history) > 1:
                simulation.stats_history = simulation.stats_history[-1:]

            frame = None
            if video_writer is not None and simulation.current_tick % video_every == 0:
                frame = renderer.render_video_frame(simulation)
                video_writer.write(frame)
            if not args.no_frames and simulation.current_tick % args.export_every == 0:
                if frame is None:
                    frame = renderer.render_video_frame(simulation)
                if pygame is None:
                    import pygame as pygame_module

                    pygame = pygame_module
                pygame.image.save(frame, frames_dir / f"frame_{simulation.current_tick:06d}.png")
            if args.checkpoint_every and simulation.current_tick % args.checkpoint_every == 0:
                save_simulation(simulation, checkpoints_dir / f"checkpoint_{simulation.current_tick:09d}.pkl")
    finally:
        if video_writer is not None:
            video_writer.close()
        metrics_logger.close()

    if renderer is not None and not args.no_maps:
        renderer.export_maps(simulation, maps_dir)
    if not args.no_analysis:
        from civ_sim.analysis import export_metrics

        export_metrics(simulation, output_dir, write_metrics_csv=False)
    if not args.no_save:
        save_simulation(simulation, save_path)
    if renderer is not None:
        renderer.shutdown()


def run_render_checkpoints(args) -> None:
    from civ_sim.render import Renderer

    checkpoint_paths = sorted(args.input.glob("checkpoint_*.pkl"))
    if args.limit > 0:
        checkpoint_paths = checkpoint_paths[: args.limit]
    if not checkpoint_paths:
        raise ValueError(f"no checkpoint_*.pkl files found in {args.input}")
    first = load_simulation(checkpoint_paths[0])
    renderer = Renderer(first.config, headless=True)
    writer = renderer.open_video_writer(args.output, fps=args.fps)
    try:
        writer.write(renderer.render_video_frame(first))
        for checkpoint_path in checkpoint_paths[1:]:
            simulation = load_simulation(checkpoint_path)
            writer.write(renderer.render_video_frame(simulation))
    finally:
        writer.close()
        renderer.shutdown()


class MetricsLogger:
    def __init__(self, path: Path, every: int):
        self.path = path
        self.every = every
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer: csv.DictWriter | None = None
        self.cumulative = {
            "cumulative_births": 0.0,
            "cumulative_deaths": 0.0,
            "cumulative_extraction_food": 0.0,
            "cumulative_extraction_wood": 0.0,
            "cumulative_extraction_stone": 0.0,
            "cumulative_delivered_food": 0.0,
            "cumulative_delivered_wood": 0.0,
            "cumulative_delivered_stone": 0.0,
            "cumulative_delivered_parts": 0.0,
            "cumulative_frontier_expansion": 0.0,
            "cumulative_storage_throughput": 0.0,
            "cumulative_workshop_throughput": 0.0,
        }

    def write(self, tick: int, stats: dict[str, Any]) -> None:
        self._update_cumulative(stats)
        if tick % self.every != 0:
            return
        row = {
            key: value
            for key, value in stats.items()
            if not isinstance(value, dict)
        }
        row["tick"] = tick
        row.update(self.cumulative)
        if self.writer is None:
            fieldnames = ["tick"] + sorted(key for key in row if key != "tick")
            self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
            self.writer.writeheader()
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def _update_cumulative(self, stats: dict[str, Any]) -> None:
        mapping = {
            "births": "cumulative_births",
            "deaths": "cumulative_deaths",
            "extraction_food": "cumulative_extraction_food",
            "extraction_wood": "cumulative_extraction_wood",
            "extraction_stone": "cumulative_extraction_stone",
            "delivered_food": "cumulative_delivered_food",
            "delivered_wood": "cumulative_delivered_wood",
            "delivered_stone": "cumulative_delivered_stone",
            "delivered_parts": "cumulative_delivered_parts",
            "frontier_expansion_rate": "cumulative_frontier_expansion",
            "storage_throughput": "cumulative_storage_throughput",
            "workshop_throughput": "cumulative_workshop_throughput",
        }
        for source, target in mapping.items():
            self.cumulative[target] += float(stats.get(source, 0.0))


class ProgressLogger:
    def __init__(self, path: Path, every: int):
        self.path = path
        self.every = every
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write(f"started_at={datetime.now(timezone.utc).isoformat()}\n")

    def write(self, simulation: Simulation, started_at: float) -> None:
        if simulation.current_tick % self.every != 0:
            return
        self._append_progress_line(simulation, started_at)

    def write_extinction_stop(self, simulation: Simulation, started_at: float) -> None:
        self._append_progress_line(simulation, started_at, prefix="stopped_on_extinction")

    def _append_progress_line(self, simulation: Simulation, started_at: float, prefix: str | None = None) -> None:
        elapsed = time.perf_counter() - started_at
        latest = simulation.stats_history[-1].to_dict() if hasattr(simulation.stats_history[-1], "to_dict") else simulation.stats_history[-1]
        fields = [
            f"tick={simulation.current_tick}",
            f"elapsed_s={elapsed:.2f}",
            f"population={latest.get('population')}",
            f"active_chunks={latest.get('active_chunks')}",
            f"roads={latest.get('road_length')}",
            f"homes={latest.get('homes')}",
            f"storage={float(latest.get('storage_throughput', 0.0)):.3f}",
            f"workshop={float(latest.get('workshop_throughput', 0.0)):.3f}",
        ]
        if prefix is not None:
            fields.insert(0, prefix)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(fields) + "\n")


class DiscordExtinctionNotifier:
    def __init__(self, webhook_url: str | None, output_dir: Path):
        self.webhook_url = webhook_url
        self.output_dir = output_dir
        self.sent = False

    @classmethod
    def from_environment(cls, output_dir: Path) -> "DiscordExtinctionNotifier":
        return cls(os.environ.get("CIV_SIM_DISCORD_WEBHOOK_URL"), output_dir)

    def notify(self, simulation: Simulation, stats, started_at: float) -> None:
        if self.sent:
            return
        self.sent = True
        elapsed = time.perf_counter() - started_at
        payload = {
            "content": (
                "civ_sim extinction: population reached 0 "
                f"at tick {simulation.current_tick:,}. "
                f"homes={stats.homes}, roads={stats.road_length}, "
                f"active_chunks={stats.active_chunks}, "
                f"elapsed={elapsed / 60.0:.1f} min, "
                f"output={self.output_dir}"
            )
        }
        if not self.webhook_url:
            self._write_status("discord_webhook_missing", payload["content"])
            return
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "civ-sim"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10) as response:
                self._write_status(f"discord_notification_sent status={response.status}", payload["content"])
        except (OSError, error.URLError, error.HTTPError) as exc:
            self._write_status(f"discord_notification_failed error={exc}", payload["content"])

    def _write_status(self, status: str, message: str) -> None:
        with (self.output_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{status} message={json.dumps(message)}\n")


def _write_run_inputs(args, config: SimConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.config and args.config.exists():
        shutil.copy2(args.config, output_dir / "config.yaml")
    else:
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_config_payload(config), handle, sort_keys=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": {
            "ticks": args.ticks,
            "seed": args.seed,
            "export_every": args.export_every,
            "video_every": args.video_every,
            "checkpoint_every": args.checkpoint_every,
            "metrics_every": args.metrics_every,
            "log_every": args.log_every,
            "load": None if args.load is None else str(args.load),
            "save": None if args.save is None else str(args.save),
            "no_video": bool(args.no_video),
            "no_frames": bool(args.no_frames),
            "no_maps": bool(args.no_maps),
            "no_analysis": bool(args.no_analysis),
            "no_save": bool(args.no_save),
            "stop_on_extinction": bool(args.stop_on_extinction),
        },
        "git_commit": _git_commit(),
        "config": _config_payload(config),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _config_payload(config: SimConfig) -> dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    main()
