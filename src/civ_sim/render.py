from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pygame

from civ_sim.analysis import district_map, frontier_map, lineage_color, lineage_map, road_map
from civ_sim.config import SimConfig
from civ_sim.constants import StructureType, TerrainType
from civ_sim.sim import Simulation


OVERLAYS = ("none", "traffic", "district", "health", "frontier", "lineage")
WorldBounds = tuple[int, int, int, int]


class Renderer:
    def __init__(self, config: SimConfig, headless: bool = False):
        self.config = config
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.display.set_caption("civ_sim")
        self.window_size = (1280, 960)
        self.screen = pygame.display.set_mode(self.window_size)
        self.font = pygame.font.SysFont("monospace", 16)
        self.tile_size = config.tile_size
        self.asset_root = Path(__file__).resolve().parents[2] / "assets" / "tiles"
        self.agent_asset_root = Path(__file__).resolve().parents[2] / "assets" / "agents"
        self.terrain_sprites = {terrain: self._load_terrain_sprite(terrain) for terrain in TerrainType}
        self.structure_sprites = {structure: self._load_structure_sprite(structure) for structure in StructureType}
        self.agent_sprite = self._load_agent_sprite()
        self.overlay_index = 0

    def cycle_overlay(self) -> str:
        self.overlay_index = (self.overlay_index + 1) % len(OVERLAYS)
        return OVERLAYS[self.overlay_index]

    def current_overlay(self) -> str:
        return OVERLAYS[self.overlay_index]

    def draw(self, simulation: Simulation) -> pygame.Surface:
        min_x, min_y, max_x, max_y = simulation.world.active_world_bounds()
        world_surface = self._render_world_surface(simulation, (min_x, min_y, max_x, max_y))
        overlay = self.current_overlay()
        if overlay != "none":
            self._draw_overlay(world_surface, simulation, overlay, min_x, min_y, max_x, max_y)

        scale = min(self.window_size[0] / world_surface.get_width(), self.window_size[1] / world_surface.get_height())
        scaled_size = (
            max(1, int(world_surface.get_width() * scale)),
            max(1, int(world_surface.get_height() * scale)),
        )
        scaled = pygame.transform.scale(world_surface, scaled_size)
        frame = pygame.Surface(self.window_size)
        frame.fill((10, 12, 15))
        frame.blit(scaled, ((self.window_size[0] - scaled_size[0]) // 2, 0))
        self._draw_hud(frame, simulation, overlay)
        self.screen.blit(frame, (0, 0))
        pygame.display.flip()
        return world_surface

    def export_frame(
        self,
        simulation: Simulation,
        path: str | Path,
        bounds: WorldBounds | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if bounds is None:
            surface = self.draw(simulation)
        else:
            surface = self._render_world_surface(simulation, bounds)
            overlay = self.current_overlay()
            if overlay != "none":
                self._draw_overlay(surface, simulation, overlay, *bounds)
        if output_size is not None and surface.get_size() != output_size:
            surface = pygame.transform.scale(surface, output_size)
        pygame.image.save(surface, path)
        return path

    def fixed_video_bounds(self) -> WorldBounds:
        width = max(1, self.config.video_view_width_tiles)
        height = max(1, self.config.video_view_height_tiles)
        min_x = self.config.video_center_x - width // 2
        min_y = self.config.video_center_y - height // 2
        return min_x, min_y, min_x + width, min_y + height

    def fixed_video_size(self) -> tuple[int, int]:
        pixels_per_tile = max(1, self.config.video_pixels_per_tile)
        return (
            max(1, self.config.video_view_width_tiles * pixels_per_tile),
            max(1, self.config.video_view_height_tiles * pixels_per_tile),
        )

    def _render_world_surface(self, simulation: Simulation, bounds: WorldBounds) -> pygame.Surface:
        min_x, min_y, max_x, max_y = bounds
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        world_surface = pygame.Surface((width * self.tile_size, height * self.tile_size))
        world_surface.fill((14, 16, 20))
        structure_draws: list[tuple[int, int, StructureType]] = []

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                terrain = simulation.world.terrain_at(x, y)
                sprite = self.terrain_sprites[terrain]
                world_surface.blit(sprite, ((x - min_x) * self.tile_size, (y - min_y) * self.tile_size))
                structure = simulation.world.get_structure(x, y)
                if structure is not None:
                    structure_draws.append((x, y, structure.kind))

        for x, y, structure in structure_draws:
            if structure != StructureType.HOME:
                self._draw_structure(world_surface, x, y, structure, min_x, min_y)
        for x, y, structure in structure_draws:
            if structure == StructureType.HOME:
                self._draw_structure(world_surface, x, y, structure, min_x, min_y)

        for agent in simulation.agents.values():
            if not (min_x <= agent.x < max_x and min_y <= agent.y < max_y):
                continue
            pixel_x = (agent.x - min_x) * self.tile_size
            pixel_y = (agent.y - min_y) * self.tile_size
            marker_rect = pygame.Rect(pixel_x + 3, pixel_y + self.tile_size - 7, self.tile_size - 6, 4)
            pygame.draw.ellipse(world_surface, lineage_color(agent.lineage_id), marker_rect)
            world_surface.blit(self.agent_sprite, (pixel_x, pixel_y))

        return world_surface

    def _draw_structure(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        structure: StructureType,
        min_x: int,
        min_y: int,
    ) -> None:
        surface.blit(
            self.structure_sprites[structure],
            ((x - min_x) * self.tile_size, (y - min_y) * self.tile_size),
        )

    def render_surface(self, simulation: Simulation) -> pygame.Surface:
        return self.draw(simulation)

    def export_video(
        self,
        frame_dir: str | Path,
        path: str | Path,
        fps: int = 20,
    ) -> Path:
        frame_dir = Path(frame_dir)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        first_frame = frame_dir / "video_frame_000001.png"
        if not first_frame.exists():
            raise ValueError("no frame paths provided for video export")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "video_frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return path

    def export_maps(self, simulation: Simulation, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_grid_map(road_map(simulation), output_dir / "road_only.png", single_channel=True)
        self._save_grid_map(district_map(simulation), output_dir / "district_map.png")
        self._save_grid_map(lineage_map(simulation), output_dir / "lineage_map.png")
        self._save_grid_map(frontier_map(simulation), output_dir / "frontier_map.png")

    def shutdown(self) -> None:
        pygame.quit()

    def _draw_overlay(
        self,
        surface: pygame.Surface,
        simulation: Simulation,
        overlay: str,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
    ) -> None:
        alpha_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                px = (x - min_x) * self.tile_size
                py = (y - min_y) * self.tile_size
                color = None
                if overlay == "traffic":
                    chunk = simulation.world.get_chunk_for_tile(x, y, activate=False)
                    local_x, local_y = simulation.world.local_coords(x, y)
                    traffic = float(chunk.traffic[local_y, local_x])
                    alpha = max(0, min(180, int(traffic * 18)))
                    color = (255, 120, 60, alpha)
                elif overlay == "health":
                    structure = simulation.world.get_structure(x, y)
                    if structure is not None:
                        health_ratio = structure.health / structure.max_health
                        color = (255 - int(health_ratio * 180), int(health_ratio * 200), 60, 130)
                elif overlay == "frontier":
                    influence = simulation.world.influence_tiles.get((x, y), 0.0)
                    alpha = max(0, min(160, int(influence * 40)))
                    color = (70, 180, 240, alpha)
                elif overlay == "lineage":
                    chunk = simulation.world.get_chunk_for_tile(x, y, activate=False)
                    local_x, local_y = simulation.world.local_coords(x, y)
                    lineage_id = int(chunk.lineage_map[local_y, local_x])
                    if lineage_id:
                        r, g, b = lineage_color(lineage_id)
                        color = (r, g, b, 120)
                elif overlay == "district":
                    terrain = simulation.world.terrain_at(x, y)
                    if terrain == TerrainType.FERTILE:
                        color = (70, 160, 70, 110)
                    elif terrain == TerrainType.FOREST:
                        color = (46, 98, 68, 110)
                    elif terrain == TerrainType.STONE:
                        color = (110, 110, 130, 110)
                if color is not None:
                    alpha_surface.fill(color, pygame.Rect(px, py, self.tile_size, self.tile_size))
        surface.blit(alpha_surface, (0, 0))

    def _draw_hud(self, frame: pygame.Surface, simulation: Simulation, overlay: str) -> None:
        if simulation.stats_history:
            latest = simulation.stats_history[-1]
            lines = [
                f"tick {simulation.current_tick}",
                f"overlay {overlay}",
                f"population {latest['population']}",
                f"homes {latest['homes']}",
                f"active_chunks {latest['active_chunks']}",
                f"roads {latest['road_length']}",
                f"births {latest['births']} deaths {latest['deaths']}",
            ]
        else:
            lines = [f"tick {simulation.current_tick}", f"overlay {overlay}"]
        y = 8
        for line in lines:
            text = self.font.render(line, True, (232, 236, 240))
            frame.blit(text, (10, y))
            y += 18

    def _terrain_sprite(self, terrain: TerrainType) -> pygame.Surface:
        palette = {
            TerrainType.GRASS: ((96, 158, 84), (122, 182, 98)),
            TerrainType.FOREST: ((42, 102, 66), (58, 124, 82)),
            TerrainType.STONE: ((120, 122, 130), (144, 146, 156)),
            TerrainType.FERTILE: ((146, 126, 68), (168, 148, 84)),
            TerrainType.WATER: ((48, 106, 166), (70, 128, 188)),
            TerrainType.HAZARD: ((130, 68, 62), (164, 86, 72)),
        }
        surface = pygame.Surface((self.tile_size, self.tile_size))
        base, accent = palette[terrain]
        surface.fill(base)
        pygame.draw.rect(surface, accent, (2, 2, self.tile_size - 4, self.tile_size - 4), border_radius=2)
        if terrain == TerrainType.FOREST:
            pygame.draw.circle(surface, (24, 76, 42), (self.tile_size // 2, self.tile_size // 2), self.tile_size // 4)
        elif terrain == TerrainType.STONE:
            pygame.draw.line(surface, (170, 172, 178), (3, 4), (self.tile_size - 3, self.tile_size - 4), 2)
        elif terrain == TerrainType.WATER:
            pygame.draw.arc(surface, (186, 220, 240), (3, 6, self.tile_size - 6, self.tile_size - 8), 0, 3.14, 2)
        elif terrain == TerrainType.HAZARD:
            pygame.draw.line(surface, (250, 212, 120), (4, 4), (self.tile_size - 4, self.tile_size - 4), 2)
        return surface

    def _load_terrain_sprite(self, terrain: TerrainType) -> pygame.Surface:
        path = self.asset_root / f"{terrain.value}.png"
        return self._load_sprite_asset(path, fallback=self._terrain_sprite(terrain))

    def _structure_sprite(self, structure: StructureType) -> pygame.Surface:
        size = self.tile_size * 2 if structure == StructureType.HOME else self.tile_size
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        if structure == StructureType.PATH:
            pygame.draw.rect(surface, (194, 176, 120, 210), (2, 6, self.tile_size - 4, self.tile_size - 12), border_radius=2)
        elif structure == StructureType.STORAGE:
            pygame.draw.rect(surface, (156, 94, 52, 220), (3, 3, self.tile_size - 6, self.tile_size - 6), border_radius=2)
        elif structure == StructureType.BEACON:
            pygame.draw.rect(surface, (224, 206, 116, 220), (6, 2, self.tile_size - 12, self.tile_size - 4), border_radius=2)
            pygame.draw.circle(surface, (255, 238, 170, 220), (self.tile_size // 2, 4), 3)
        elif structure == StructureType.WALL:
            pygame.draw.rect(surface, (112, 114, 126, 230), (1, 4, self.tile_size - 2, self.tile_size - 8))
        elif structure == StructureType.GATE:
            pygame.draw.rect(surface, (132, 92, 60, 230), (3, 2, self.tile_size - 6, self.tile_size - 4))
            pygame.draw.rect(surface, (70, 42, 22, 230), (6, 4, self.tile_size - 12, self.tile_size - 8))
        elif structure == StructureType.HOME:
            pygame.draw.rect(surface, (92, 138, 72), (0, 0, size, size))
            pygame.draw.rect(surface, (150, 108, 62), (3, 22, size - 6, 7))
            pygame.draw.polygon(surface, (130, 58, 34), [(4, 16), (size // 2, 4), (size - 4, 16)])
            pygame.draw.rect(surface, (194, 122, 62), (7, 15, size - 14, 12))
            pygame.draw.rect(surface, (76, 49, 31), (14, 20, 5, 8))
        elif structure == StructureType.WORKSHOP:
            pygame.draw.rect(surface, (104, 112, 82, 230), (2, 3, self.tile_size - 4, self.tile_size - 6))
            pygame.draw.circle(surface, (196, 206, 180, 230), (self.tile_size // 2, self.tile_size // 2), 3)
        return surface

    def _load_structure_sprite(self, structure: StructureType) -> pygame.Surface:
        path = self.asset_root / f"{structure.value}.png"
        expected_size = (self.tile_size * 2, self.tile_size * 2) if structure == StructureType.HOME else None
        return self._load_sprite_asset(path, fallback=self._structure_sprite(structure), expected_size=expected_size)

    def _load_sprite_asset(
        self,
        path: Path,
        fallback: pygame.Surface,
        expected_size: tuple[int, int] | None = None,
    ) -> pygame.Surface:
        if not path.exists():
            return fallback
        surface = pygame.image.load(path).convert_alpha()
        target_size = expected_size or (self.tile_size, self.tile_size)
        if surface.get_size() != target_size:
            surface = pygame.transform.scale(surface, target_size)
        return surface

    def _agent_placeholder(self) -> pygame.Surface:
        surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        pygame.draw.circle(surface, (40, 42, 48, 220), (self.tile_size // 2, self.tile_size // 2 + 1), self.tile_size // 3)
        pygame.draw.circle(surface, (220, 224, 230, 240), (self.tile_size // 2, self.tile_size // 2 - 2), self.tile_size // 4)
        return surface

    def _load_agent_sprite(self) -> pygame.Surface:
        path = self.agent_asset_root / "agent_base.png"
        return self._load_sprite_asset(path, fallback=self._agent_placeholder())

    def _save_grid_map(self, grid, path: Path, single_channel: bool = False) -> None:
        height, width = grid.shape[:2]
        surface = pygame.Surface((width, height))
        if single_channel:
            for y in range(height):
                for x in range(width):
                    value = int(grid[y, x])
                    surface.set_at((x, y), (value, value, value))
        else:
            for y in range(height):
                for x in range(width):
                    surface.set_at((x, y), tuple(int(v) for v in grid[y, x]))
        pygame.image.save(surface, path)
