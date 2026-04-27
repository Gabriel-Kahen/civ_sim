from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pygame

from civ_sim.analysis import export_metrics
from civ_sim.config import SimConfig
from civ_sim.io import load_simulation, save_simulation
from civ_sim.render import Renderer
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
    experiment.add_argument("--save", type=Path, default=Path("exports/run/final_state.json"))

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


def _load_or_create(args) -> Simulation:
    if args.load:
        return load_simulation(args.load)
    config = SimConfig.from_yaml(args.config) if args.config and args.config.exists() else SimConfig()
    config.seed = args.seed
    config.export_every = args.export_every
    return Simulation.create(config)


def run_sandbox(args) -> None:
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
    simulation = _load_or_create(args)
    renderer = Renderer(simulation.config, headless=True)
    output_dir = args.output
    frames_dir = output_dir / "frames"
    maps_dir = output_dir / "maps"
    video_frames_dir = output_dir / ".video_frames"
    video_path = output_dir / "animation.mp4"
    frames_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    video_bounds = renderer.fixed_video_bounds()
    video_size = renderer.fixed_video_size()

    while simulation.current_tick < args.ticks:
        simulation.step()
        video_frame_path = video_frames_dir / f"video_frame_{simulation.current_tick:06d}.png"
        renderer.export_frame(simulation, video_frame_path, bounds=video_bounds, output_size=video_size)
        if simulation.current_tick % args.export_every == 0:
            renderer.export_frame(
                simulation,
                frames_dir / f"frame_{simulation.current_tick:06d}.png",
                bounds=video_bounds,
                output_size=video_size,
            )

    renderer.export_video(video_frames_dir, video_path, fps=20)
    renderer.export_maps(simulation, maps_dir)
    export_metrics(simulation, output_dir)
    save_simulation(simulation, args.save)
    renderer.shutdown()
    shutil.rmtree(video_frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
