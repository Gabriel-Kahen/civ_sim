from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from civ_sim.config import SimConfig
from civ_sim.constants import (
    ActionType,
    CARDINAL_DIRECTIONS,
    DIRECTION_VECTORS,
    HARVESTABLE_TERRAIN_TO_RESOURCE,
    RESOURCE_ORDER,
    RESOURCE_INDEX,
    STRUCTURE_INDEX,
    STRUCTURE_ORDER,
    TERRAIN_ORDER,
    Direction,
    ResourceType,
    StructureType,
    TerrainType,
)
from civ_sim.controller import ControllerRuntime, FOUNDER_ARCHETYPES, TRAIT_ORDER
from civ_sim.models import Agent, AgentGenome, Inventory, SimulationStats
from civ_sim.world import World


PATCH_CHANNELS = 22
SELF_FEATURES = 14
TERRAIN_ONE_HOT = tuple(
    tuple(1.0 if index == active else 0.0 for index in range(len(TERRAIN_ORDER)))
    for active in range(len(TERRAIN_ORDER))
)
STRUCTURE_ZERO = (0.0,) * len(STRUCTURE_ORDER)
STRUCTURE_ONE_HOT = {
    structure: tuple(1.0 if index == STRUCTURE_INDEX[structure] else 0.0 for index in range(len(STRUCTURE_ORDER)))
    for structure in STRUCTURE_ORDER
}
CARDINAL_OFFSETS = tuple(DIRECTION_VECTORS[direction] for direction in CARDINAL_DIRECTIONS)


@dataclass(slots=True)
class Simulation:
    config: SimConfig
    world: World
    controller: ControllerRuntime
    rng: random.Random
    agents: dict[int, Agent]
    current_tick: int = 0
    next_agent_id: int = 1
    next_lineage_id: int = 1
    stats_history: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.stats_history is None:
            self.stats_history = []

    @classmethod
    def create(cls, config: SimConfig | None = None) -> "Simulation":
        config = config or SimConfig()
        observation_size = (
            config.observation_diameter * config.observation_diameter * PATCH_CHANNELS
            + SELF_FEATURES
            + len(TRAIT_ORDER)
        )
        world = World(config)
        controller = ControllerRuntime(config, observation_size)
        sim = cls(
            config=config,
            world=world,
            controller=controller,
            rng=random.Random(config.seed),
            agents={},
        )
        sim.reset()
        return sim

    def reset(self) -> None:
        torch.manual_seed(self.config.seed)
        self.world = World(self.config)
        self.world.initialize()
        self.agents = {}
        self.current_tick = 0
        self.next_agent_id = 1
        self.next_lineage_id = 1
        self.stats_history = []
        self._bootstrap_world()
        self.world.recompute_influence(set(self.agent_positions()))

    def snapshot(self) -> dict[str, Any]:
        config_payload = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(self.config).items()
        }
        return {
            "config": config_payload,
            "current_tick": self.current_tick,
            "next_agent_id": self.next_agent_id,
            "next_lineage_id": self.next_lineage_id,
            "world": self.world.snapshot(),
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "stats_history": self.stats_history,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "Simulation":
        config = SimConfig.from_dict(payload["config"])
        observation_size = (
            config.observation_diameter * config.observation_diameter * PATCH_CHANNELS
            + SELF_FEATURES
            + len(TRAIT_ORDER)
        )
        world = World.from_snapshot(config, payload["world"])
        controller = ControllerRuntime(config, observation_size)
        sim = cls(
            config=config,
            world=world,
            controller=controller,
            rng=random.Random(config.seed),
            agents={},
        )
        sim.current_tick = int(payload["current_tick"])
        sim.next_agent_id = int(payload["next_agent_id"])
        sim.next_lineage_id = int(payload["next_lineage_id"])
        sim.agents = {agent.agent_id: agent for agent in (Agent.from_dict(item) for item in payload["agents"])}
        sim.stats_history = list(payload.get("stats_history", []))
        return sim

    def agent_positions(self) -> list[tuple[int, int]]:
        return [(agent.x, agent.y) for agent in self.agents.values()]

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def step(self) -> SimulationStats:
        self.current_tick += 1
        self.world.update_resources()
        self.world.decay_buffers()
        structure_events = self._update_structures_and_homes()
        influence_events = self.world.recompute_influence(set(self.agent_positions()))

        births = structure_events["births"]
        deaths = 0
        extraction = defaultdict(float)
        deliveries = defaultdict(float)
        action_order = list(self.agents.keys())
        self.rng.shuffle(action_order)
        occupancy = Counter((agent.x, agent.y) for agent in self.agents.values())

        for agent_id in action_order:
            agent = self.agents.get(agent_id)
            if agent is None:
                continue
            died = self._metabolize_agent(agent)
            if died:
                deaths += 1
                continue
            acted, local_births, local_deaths, local_extraction, local_deliveries = self._step_agent(agent, occupancy)
            births += local_births
            deaths += local_deaths
            for key, value in local_extraction.items():
                extraction[key] += value
            for key, value in local_deliveries.items():
                deliveries[key] += value
            if acted:
                occupancy[(agent.x, agent.y)] += 1

        stats = self._collect_stats(
            births=births,
            deaths=deaths + structure_events["deaths"],
            extraction=extraction,
            deliveries=deliveries,
            frontier_activated=influence_events["activated"],
        )
        self.stats_history.append(stats.to_dict())
        return stats

    def _bootstrap_world(self) -> None:
        home_positions: list[tuple[int, int]] = []
        spacing = self.config.initial_home_spacing
        if self.config.initial_home_count <= 1:
            home_positions.append((0, 0))
        else:
            start = -((self.config.initial_home_count - 1) * spacing) // 2
            for index in range(self.config.initial_home_count):
                x = start + index * spacing
                y = 0
                if self.config.initial_home_count >= 3 and index != self.config.initial_home_count // 2:
                    y = spacing
                home_positions.append((x, y))

        archetypes = tuple(FOUNDER_ARCHETYPES.keys())
        homes = []
        for index, (x, y) in enumerate(home_positions):
            lineage_id = self.next_lineage_id
            home = self.world.add_structure(StructureType.HOME, x, y, lineage_id=lineage_id, tick=0)
            home.inventory.food = self.config.seed_home_food
            home.inventory.wood = self.config.seed_home_wood
            home.inventory.stone = self.config.seed_home_stone
            home.inventory.parts = self.config.seed_home_parts
            homes.append(home)
            self.next_lineage_id += 1

            for dx, dy, resource in (
                (1, 0, ResourceType.FOOD),
                (-1, 0, ResourceType.WOOD),
                (0, 1, ResourceType.STONE),
                (0, -1, ResourceType.PARTS),
            ):
                self.world.add_ground_resource(x + dx, y + dy, resource, self.config.seed_ground_cache)

            per_home = max(1, self.config.initial_agents // max(self.config.initial_home_count, 1))
            for founder_index in range(per_home):
                spawn_positions = self._adjacent_open_tiles(x, y)
                if not spawn_positions:
                    break
                spawn_x, spawn_y = self.rng.choice(spawn_positions)
                archetype = archetypes[(index + founder_index) % len(archetypes)]
                genome = self.controller.mutate_genome(
                    self.controller.create_founder_genome(archetype),
                    strength=0.025,
                )
                self._spawn_agent(
                    x=spawn_x,
                    y=spawn_y,
                    lineage_id=lineage_id,
                    home_id=home.structure_id,
                    genome=genome,
                    energy=self.config.founder_energy_min + self.rng.random() * self.config.founder_energy_random,
                    generation=0,
                    starting_carried_resource=ResourceType.PARTS if archetype in {"builder", "mason"} else None,
                    starting_carried_amount=self.config.founder_carried_parts if archetype in {"builder", "mason"} else 0.0,
                )

    def _spawn_agent(
        self,
        x: int,
        y: int,
        lineage_id: int,
        home_id: int | None,
        genome: AgentGenome,
        energy: float,
        generation: int,
        starting_carried_resource: ResourceType | None = None,
        starting_carried_amount: float = 0.0,
    ) -> Agent:
        agent = Agent(
            agent_id=self.next_agent_id,
            lineage_id=lineage_id,
            x=x,
            y=y,
            home_id=home_id,
            age=0,
            max_age=self.rng.randint(self.config.agent_min_lifespan, self.config.agent_max_lifespan),
            energy=energy,
            carried_resource=None if starting_carried_resource is None else starting_carried_resource.value,
            carried_amount=starting_carried_amount,
            hidden_state=torch.zeros(self.controller.hidden_size, dtype=torch.float32),
            genome=genome,
            birth_tick=self.current_tick,
            generation=generation,
        )
        self.agents[agent.agent_id] = agent
        self.next_agent_id += 1
        return agent

    def _step_agent(
        self,
        agent: Agent,
        occupancy: Counter,
    ) -> tuple[bool, int, int, dict[ResourceType, float], dict[ResourceType, float]]:
        local_births = 0
        local_deaths = 0
        extraction: dict[ResourceType, float] = defaultdict(float)
        deliveries: dict[ResourceType, float] = defaultdict(float)
        self._opportunistic_deposit(agent, deliveries)

        speed = self.controller.move_speed(agent.genome)
        load_penalty = 1.0 + (agent.carried_amount / max(self.controller.carry_capacity(agent.genome), 1e-6)) * 0.6
        tile_penalty = self.world.move_cost_multiplier(agent.x, agent.y)
        agent.action_progress = min(2.0, agent.action_progress + speed / (load_penalty * tile_penalty))
        if agent.action_progress < 1.0:
            return False, 0, 0, extraction, deliveries
        agent.action_progress -= 1.0

        observation = self._build_observation(agent, occupancy)
        hidden_state = (
            agent.hidden_state
            if isinstance(agent.hidden_state, torch.Tensor)
            else torch.tensor(agent.hidden_state, dtype=torch.float32)
        )
        controller_output = self.controller.forward(agent.genome, observation, hidden_state)
        action, direction = self._select_action(agent, controller_output, occupancy)
        agent.hidden_state = controller_output.hidden_state
        moved, born, died, extraction, deliveries = self._execute_action(agent, action, direction)
        local_births += born
        local_deaths += died
        return moved, local_births, local_deaths, extraction, deliveries

    def _metabolize_agent(self, agent: Agent) -> bool:
        agent.age += 1
        energy_decay = self.config.base_energy_decay + agent.carried_amount * self.config.carried_energy_decay
        if self.current_tick - agent.birth_tick < self.config.founder_support_ticks:
            energy_decay *= self.config.founder_energy_decay_scale
        agent.energy -= energy_decay
        hazard_damage = self.world.hazard_at(agent.x, agent.y) * self.config.hazard_damage_scale
        if hazard_damage > 0.0:
            agent.energy -= hazard_damage * max(0.4, 1.2 - agent.genome.traits["risk_tolerance"])

        if (
            agent.energy < self.config.carried_food_eat_threshold
            and agent.carried_resource == ResourceType.FOOD.value
            and agent.carried_amount >= 1.0
        ):
            agent.carried_amount -= 1.0
            agent.energy += self.config.food_energy_gain
            if agent.carried_amount <= 0.0:
                agent.carried_amount = 0.0
                agent.carried_resource = None

        structure = self.world.get_structure(agent.x, agent.y)
        if (
            agent.energy < self.config.structure_food_eat_threshold
            and structure is not None
            and structure.inventory is not None
            and structure.inventory.food >= 1.0
        ):
            structure.inventory.remove(ResourceType.FOOD, 1.0)
            agent.energy += self.config.food_energy_gain

        support_home = self._home_structure_for_agent(agent)
        if (
            support_home is not None
            and support_home.inventory is not None
            and abs(self._world_position_of_structure(support_home)[0] - agent.x) + abs(self._world_position_of_structure(support_home)[1] - agent.y)
            <= self.config.home_support_radius
            and agent.energy < self.config.home_support_energy_threshold
            and support_home.inventory.food
            >= self.config.home_support_food_reserve + self.config.home_support_food_amount
        ):
            support_home.inventory.remove(ResourceType.FOOD, self.config.home_support_food_amount)
            agent.energy += self.config.food_energy_gain * self.config.home_support_food_amount

        agent.energy = min(agent.energy, self.config.max_energy)

        if agent.energy <= 0.0 or agent.age >= agent.max_age:
            self._kill_agent(agent.agent_id)
            return True
        return False

    def _kill_agent(self, agent_id: int) -> None:
        agent = self.agents.pop(agent_id, None)
        if agent is None:
            return
        if agent.carried_resource is not None and agent.carried_amount > 0.0:
            self.world.add_ground_resource(agent.x, agent.y, ResourceType(agent.carried_resource), agent.carried_amount)

    def _build_observation(self, agent: Agent, occupancy: Counter) -> torch.Tensor:
        radius = self.config.observation_radius
        vision_range = self.controller.vision_range(agent.genome)
        patch: list[float] = []
        world = self.world
        chunk_size = world.config.chunk_size
        chunk_for_key = world._chunk_for_key
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) > vision_range:
                    patch.extend([0.0] * PATCH_CHANNELS)
                    continue
                tx = agent.x + dx
                ty = agent.y + dy
                chunk_x, local_x = divmod(tx, chunk_size)
                chunk_y, local_y = divmod(ty, chunk_size)
                chunk = chunk_for_key(chunk_x, chunk_y, activate=False)
                terrain_index = int(chunk.terrain[local_y, local_x])
                structure = chunk.structures.get((local_x, local_y))
                resource_amount = float(chunk.resource_amount[local_y, local_x]) / 20.0
                resource_quality = float(chunk.resource_quality[local_y, local_x]) / 2.0
                hazard = float(chunk.hazard[local_y, local_x])
                traffic = float(chunk.traffic[local_y, local_x]) / 10.0
                influence = world.influence_tiles.get((tx, ty), 0.0) / 5.0
                ground = chunk.ground_resources[local_y, local_x]
                ground_total = float(ground[0] + ground[1] + ground[2] + ground[3]) / 10.0
                structure_health = 0.0 if structure is None else structure.health / structure.max_health
                structure_inventory = 0.0 if structure is None or structure.inventory is None else structure.inventory.total() / 40.0
                patch.extend(TERRAIN_ONE_HOT[terrain_index])
                patch.extend(STRUCTURE_ZERO if structure is None else STRUCTURE_ONE_HOT[structure.kind])
                patch.extend(
                    [
                        min(resource_amount, 1.0),
                        min(resource_quality, 1.0),
                        min(occupancy[(tx, ty)] / 4.0, 1.0),
                        min(hazard, 1.0),
                        min(traffic, 1.0),
                        min(influence, 1.0),
                        min(ground_total, 1.0),
                        min(structure_health, 1.0),
                        min(structure_inventory, 1.0),
                    ]
                )

        carried_one_hot = [0.0] * (len(RESOURCE_ORDER) + 1)
        if agent.carried_resource is None:
            carried_one_hot[-1] = 1.0
        else:
            carried_one_hot[RESOURCE_INDEX[ResourceType(agent.carried_resource)]] = 1.0
        current_chunk_x, current_local_x = divmod(agent.x, chunk_size)
        current_chunk_y, current_local_y = divmod(agent.y, chunk_size)
        current_chunk = chunk_for_key(current_chunk_x, current_chunk_y, activate=False)
        current_structure = current_chunk.structures.get((current_local_x, current_local_y))
        structure_health = 0.0 if current_structure is None else current_structure.health / current_structure.max_health
        nearby_agents = sum(
            occupancy[(agent.x + dx, agent.y + dy)]
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            if not (dx == 0 and dy == 0)
        )
        self_features = [
            min(agent.energy / 20.0, 1.5),
            min(agent.carried_amount / max(self.controller.carry_capacity(agent.genome), 1e-6), 1.0),
            structure_health,
            min(nearby_agents / 8.0, 1.0),
            min(abs(agent.x) / 80.0, 1.0),
            min(abs(agent.y) / 80.0, 1.0),
            min(world.influence_tiles.get((agent.x, agent.y), 0.0) / 5.0, 1.0),
            min(float(current_chunk.hazard[current_local_y, current_local_x]), 1.0),
            min(
                float(
                    current_chunk.ground_resources[current_local_y, current_local_x, 0]
                    + current_chunk.ground_resources[current_local_y, current_local_x, 1]
                    + current_chunk.ground_resources[current_local_y, current_local_x, 2]
                    + current_chunk.ground_resources[current_local_y, current_local_x, 3]
                )
                / 10.0,
                1.0,
            ),
        ]
        values = patch + carried_one_hot + self_features + self.controller.trait_vector(agent.genome)
        return torch.tensor(values, dtype=torch.float32)

    def _select_action(
        self,
        agent: Agent,
        controller_output,
        occupancy: Counter,
    ) -> tuple[ActionType, Direction]:
        legal_actions = self._legal_action_mask(agent)
        action_logits = controller_output.action_logits.clone()
        for action_type, legal in legal_actions.items():
            if not legal:
                action_logits[int(action_type)] = -1e9
        action = self._sample_from_logits(action_logits, len(ActionType))

        legal_directions = self._legal_direction_mask(agent, action)
        direction_logits = controller_output.direction_logits.clone()
        for direction, legal in legal_directions.items():
            if not legal:
                direction_logits[int(direction)] = -1e9
        direction = self._sample_from_logits(direction_logits, len(Direction))

        if action == ActionType.MOVE and agent.stuck_ticks >= self.config.anti_stuck_ticks:
            valid = [direction for direction, allowed in legal_directions.items() if allowed]
            if valid:
                direction = self.rng.choice(valid)
                agent.stuck_ticks = 0
        return action, direction

    def _sample_from_logits(self, logits: torch.Tensor, count: int):
        values = np.asarray(logits.detach().cpu().tolist(), dtype=np.float64)
        maximum = float(np.max(values))
        if maximum < -1e8:
            enum_type = ActionType if count == len(ActionType) else Direction
            return enum_type(0)
        weights = np.exp(np.clip(values - maximum, -50.0, 50.0))
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            enum_type = ActionType if count == len(ActionType) else Direction
            return enum_type(int(np.argmax(values)))
        choice = self.rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += float(weight)
            if cumulative >= choice:
                enum_type = ActionType if count == len(ActionType) else Direction
                return enum_type(index)
        enum_type = ActionType if count == len(ActionType) else Direction
        return enum_type(int(np.argmax(values)))

    def _legal_action_mask(self, agent: Agent) -> dict[ActionType, bool]:
        structure = self.world.get_structure(agent.x, agent.y)
        harvestable = self.world.terrain_at(agent.x, agent.y) in HARVESTABLE_TERRAIN_TO_RESOURCE and (
            self.world.resource_amount_at(agent.x, agent.y) > 0.2
        )
        ground_resources = self.world.ground_resource_vector(agent.x, agent.y)
        can_carry_more = agent.carried_amount < self.controller.carry_capacity(agent.genome)
        carrying_parts = agent.carried_resource == ResourceType.PARTS.value and agent.carried_amount >= 1.0
        accessible_inventory = self._accessible_inventory_structures(agent)
        has_inventory = bool(accessible_inventory)
        has_withdrawable_inventory = any(structure.inventory is not None and structure.inventory.total() > 0.0 for structure in accessible_inventory)

        has_build_site = any(self._can_build_on(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS)
        has_damaged_neighbor = any(
            (structure := self.world.get_structure(agent.x + dx, agent.y + dy)) is not None
            and structure.health < structure.max_health * 0.98
            for dx, dy in CARDINAL_OFFSETS
        )

        mask = {
            ActionType.MOVE: any(
                self.world.can_enter_tile(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS
            ),
            ActionType.HARVEST: harvestable and can_carry_more,
            ActionType.PICKUP: bool(np.any(ground_resources > 0.0)) and can_carry_more,
            ActionType.DROP: agent.carried_resource is not None and agent.carried_amount > 0.0,
            ActionType.DEPOSIT: has_inventory and agent.carried_resource is not None,
            ActionType.WITHDRAW: has_withdrawable_inventory and can_carry_more,
            ActionType.BUILD_PATH: carrying_parts and any(
                self._can_build_path(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS
            ),
            ActionType.BUILD_STORAGE: carrying_parts and any(
                self._can_build_storage(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS
            ),
            ActionType.BUILD_BEACON: carrying_parts and any(
                self._can_build_beacon(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS
            ),
            ActionType.BUILD_WALL: carrying_parts and any(
                self._can_build_wall(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS
            ),
            ActionType.BUILD_GATE: carrying_parts and any(
                self._can_build_gate(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS
            ),
            ActionType.BUILD_HOME: carrying_parts and any(
                self._can_build_home(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS
            ),
            ActionType.BUILD_WORKSHOP: carrying_parts and any(
                self._can_build_workshop(agent.x + dx, agent.y + dy) for dx, dy in CARDINAL_OFFSETS
            ),
            ActionType.REPAIR: carrying_parts and has_damaged_neighbor,
            ActionType.IDLE: True,
            ActionType.GUARD: True,
        }
        return mask

    def _legal_direction_mask(self, agent: Agent, action: ActionType) -> dict[Direction, bool]:
        if action == ActionType.MOVE:
            return {
                Direction.CENTER: False,
                Direction.NORTH: self.world.can_enter_tile(agent.x, agent.y - 1),
                Direction.EAST: self.world.can_enter_tile(agent.x + 1, agent.y),
                Direction.SOUTH: self.world.can_enter_tile(agent.x, agent.y + 1),
                Direction.WEST: self.world.can_enter_tile(agent.x - 1, agent.y),
            }

        if action in {
            ActionType.BUILD_PATH,
            ActionType.BUILD_STORAGE,
            ActionType.BUILD_BEACON,
            ActionType.BUILD_WALL,
            ActionType.BUILD_GATE,
            ActionType.BUILD_HOME,
            ActionType.BUILD_WORKSHOP,
            ActionType.REPAIR,
        }:
            mask = {Direction.CENTER: False}
            for direction, (dx, dy) in DIRECTION_VECTORS.items():
                if direction == Direction.CENTER:
                    continue
                tx = agent.x + dx
                ty = agent.y + dy
                if action == ActionType.REPAIR:
                    structure = self.world.get_structure(tx, ty)
                    mask[direction] = structure is not None and structure.health < structure.max_health * 0.98
                elif action == ActionType.BUILD_PATH:
                    mask[direction] = self._can_build_path(tx, ty)
                elif action == ActionType.BUILD_STORAGE:
                    mask[direction] = self._can_build_storage(tx, ty)
                elif action == ActionType.BUILD_BEACON:
                    mask[direction] = self._can_build_beacon(tx, ty)
                elif action == ActionType.BUILD_WALL:
                    mask[direction] = self._can_build_wall(tx, ty)
                elif action == ActionType.BUILD_GATE:
                    mask[direction] = self._can_build_gate(tx, ty)
                elif action == ActionType.BUILD_HOME:
                    mask[direction] = self._can_build_home(tx, ty)
                elif action == ActionType.BUILD_WORKSHOP:
                    mask[direction] = self._can_build_workshop(tx, ty)
                else:
                    mask[direction] = self._can_build_on(tx, ty)
            return mask

        if action in {ActionType.DEPOSIT, ActionType.WITHDRAW}:
            mask = {}
            for direction, (dx, dy) in DIRECTION_VECTORS.items():
                structure = self.world.get_structure(agent.x + dx, agent.y + dy)
                has_inventory = structure is not None and structure.inventory is not None
                if action == ActionType.WITHDRAW:
                    has_inventory = has_inventory and structure.inventory.total() > 0.0
                mask[direction] = has_inventory
            return mask

        return {
            Direction.CENTER: True,
            Direction.NORTH: False,
            Direction.EAST: False,
            Direction.SOUTH: False,
            Direction.WEST: False,
        }

    def _execute_action(
        self,
        agent: Agent,
        action: ActionType,
        direction: Direction,
    ) -> tuple[bool, int, int, dict[ResourceType, float], dict[ResourceType, float]]:
        extraction: dict[ResourceType, float] = defaultdict(float)
        deliveries: dict[ResourceType, float] = defaultdict(float)
        moved = False
        births = 0
        deaths = 0
        dx, dy = DIRECTION_VECTORS[direction]
        target_x = agent.x + dx
        target_y = agent.y + dy

        if (
            action == ActionType.MOVE
            and direction != Direction.CENTER
            and self.world.can_enter_tile(target_x, target_y)
        ):
            previous = (agent.x, agent.y)
            self.world.activate_tile(target_x, target_y)
            agent.x = target_x
            agent.y = target_y
            moved = previous != (agent.x, agent.y)
            if moved:
                self.world.mark_traffic(agent.x, agent.y, agent.lineage_id)
                agent.last_move_tick = self.current_tick
                agent.stuck_ticks = 0
            else:
                agent.stuck_ticks += 1
        elif action == ActionType.HARVEST:
            terrain = self.world.terrain_at(agent.x, agent.y)
            if terrain in HARVESTABLE_TERRAIN_TO_RESOURCE:
                resource = HARVESTABLE_TERRAIN_TO_RESOURCE[terrain]
                if agent.carried_resource in (None, resource.value):
                    capacity = self.controller.carry_capacity(agent.genome) - agent.carried_amount
                    amount = min(
                        self.config.harvest_amount * self.world.resource_quality_at(agent.x, agent.y),
                        self.world.resource_amount_at(agent.x, agent.y),
                        capacity,
                    )
                    if amount > 0.0:
                        self.world.modify_resource(agent.x, agent.y, -amount)
                        agent.carried_resource = resource.value
                        agent.carried_amount += amount
                        extraction[resource] += amount
        elif action == ActionType.PICKUP:
            available = self.world.ground_resource_vector(agent.x, agent.y)
            best_index = int(np.argmax(available))
            if available[best_index] > 0.0:
                resource = RESOURCE_ORDER[best_index]
                if agent.carried_resource in (None, resource.value):
                    capacity = self.controller.carry_capacity(agent.genome) - agent.carried_amount
                    amount = min(self.config.pickup_amount, available[best_index], capacity)
                    taken = self.world.take_ground_resource(agent.x, agent.y, resource, amount)
                    if taken > 0.0:
                        agent.carried_resource = resource.value
                        agent.carried_amount += taken
        elif action == ActionType.DROP and agent.carried_resource is not None and agent.carried_amount > 0.0:
            amount = min(self.config.deposit_amount, agent.carried_amount)
            self.world.add_ground_resource(agent.x, agent.y, ResourceType(agent.carried_resource), amount)
            agent.carried_amount -= amount
            if agent.carried_amount <= 0.0:
                agent.carried_amount = 0.0
                agent.carried_resource = None
        elif action == ActionType.DEPOSIT and agent.carried_resource is not None:
            structure = self._deposit_structure_for_direction(agent, direction)
            if structure is not None and structure.inventory is not None:
                amount = min(self.config.deposit_amount, agent.carried_amount)
                structure.inventory.add(ResourceType(agent.carried_resource), amount)
                structure.stored_throughput += amount
                deliveries[ResourceType(agent.carried_resource)] += amount
                agent.carried_amount -= amount
                if agent.carried_amount <= 0.0:
                    agent.carried_amount = 0.0
                    agent.carried_resource = None
        elif action == ActionType.WITHDRAW:
            structure = self._inventory_structure_for_direction(agent, direction)
            if structure is not None and structure.inventory is not None:
                resource = self._choose_withdraw_resource(agent, structure.inventory)
                if resource is not None and agent.carried_resource in (None, resource.value):
                    capacity = self.controller.carry_capacity(agent.genome) - agent.carried_amount
                    amount = min(self.config.withdraw_amount, capacity)
                    taken = structure.inventory.remove(resource, amount)
                    if taken > 0.0:
                        agent.carried_resource = resource.value
                        agent.carried_amount += taken
        elif action in {
            ActionType.BUILD_PATH,
            ActionType.BUILD_STORAGE,
            ActionType.BUILD_BEACON,
            ActionType.BUILD_WALL,
            ActionType.BUILD_GATE,
            ActionType.BUILD_HOME,
            ActionType.BUILD_WORKSHOP,
        }:
            structure_kind = {
                ActionType.BUILD_PATH: StructureType.PATH,
                ActionType.BUILD_STORAGE: StructureType.STORAGE,
                ActionType.BUILD_BEACON: StructureType.BEACON,
                ActionType.BUILD_WALL: StructureType.WALL,
                ActionType.BUILD_GATE: StructureType.GATE,
                ActionType.BUILD_HOME: StructureType.HOME,
                ActionType.BUILD_WORKSHOP: StructureType.WORKSHOP,
            }[action]
            if (
                direction != Direction.CENTER
                and agent.carried_resource == ResourceType.PARTS.value
                and (
                    (structure_kind == StructureType.PATH and self._can_build_path(target_x, target_y))
                    or (structure_kind == StructureType.STORAGE and self._can_build_storage(target_x, target_y))
                    or (structure_kind == StructureType.BEACON and self._can_build_beacon(target_x, target_y))
                    or (structure_kind == StructureType.WALL and self._can_build_wall(target_x, target_y))
                    or (structure_kind == StructureType.GATE and self._can_build_gate(target_x, target_y))
                    or (structure_kind == StructureType.HOME and self._can_build_home(target_x, target_y))
                    or (structure_kind == StructureType.WORKSHOP and self._can_build_workshop(target_x, target_y))
                )
            ):
                cost = self.config.structure_build_cost(structure_kind, ResourceType.PARTS)
                if agent.carried_amount >= cost:
                    structure = self.world.add_structure(structure_kind, target_x, target_y, agent.lineage_id, self.current_tick)
                    structure.health = structure.max_health * self.config.build_health_fraction
                    if structure.inventory is not None and structure.kind == StructureType.HOME:
                        structure.inventory.food = self.config.new_home_food
                    agent.carried_amount -= cost
                    if agent.carried_amount <= 0.0:
                        agent.carried_amount = 0.0
                        agent.carried_resource = None
        elif action == ActionType.REPAIR and agent.carried_resource == ResourceType.PARTS.value:
            structure = self.world.get_structure(target_x, target_y)
            if structure is not None:
                cost = self.config.structure_repair_cost(structure.kind, ResourceType.PARTS)
                if agent.carried_amount >= cost:
                    structure.health = min(
                        structure.max_health,
                        structure.health + structure.max_health * self.config.repair_health_fraction,
                    )
                    agent.carried_amount -= cost
                    if agent.carried_amount <= 0.0:
                        agent.carried_amount = 0.0
                        agent.carried_resource = None
        elif action == ActionType.GUARD:
            for dx, dy in CARDINAL_OFFSETS:
                structure = self.world.get_structure(agent.x + dx, agent.y + dy)
                if structure is not None and structure.kind in {
                    StructureType.WALL,
                    StructureType.GATE,
                    StructureType.HOME,
                    StructureType.STORAGE,
                }:
                    structure.health = min(structure.max_health, structure.health + self.config.guard_repair_bonus)

        if action != ActionType.MOVE:
            agent.stuck_ticks += 1

        if agent.energy <= 0.0:
            self._kill_agent(agent.agent_id)
            deaths += 1
        return moved, births, deaths, extraction, deliveries

    def _update_structures_and_homes(self) -> dict[str, int]:
        births = 0
        deaths = 0
        agents_by_home = defaultdict(list)
        for agent in self.agents.values():
            if agent.home_id is not None:
                agents_by_home[agent.home_id].append(agent)

        for (world_x, world_y), structure in list(self.world.iter_active_structures()):
            structure.stored_throughput *= 0.92
            structure.processed_throughput *= 0.92
            structure.health -= self.config.structure_decay_rate(structure.kind)
            if structure.inventory is not None and structure.kind == StructureType.WORKSHOP:
                self._pull_workshop_inputs(world_x, world_y, structure.inventory)
                craft_batches = min(
                    structure.inventory.wood / self.config.workshop_wood_input,
                    structure.inventory.stone / self.config.workshop_stone_input,
                )
                craft_batches = min(craft_batches, 1.0)
                if craft_batches > 0.0:
                    structure.inventory.wood -= craft_batches * self.config.workshop_wood_input
                    structure.inventory.stone -= craft_batches * self.config.workshop_stone_input
                    output = craft_batches * self.config.workshop_parts_output
                    structure.inventory.parts += output
                    structure.processed_throughput += output

            if structure.inventory is not None and structure.kind == StructureType.HOME:
                upkeep_taken = structure.inventory.remove(ResourceType.FOOD, self.config.home_food_upkeep)
                if upkeep_taken < self.config.home_food_upkeep:
                    structure.health -= self.config.home_starvation_damage
                if self._access_score(world_x, world_y) < 1.0:
                    structure.health -= self.config.home_low_access_damage
                residents = agents_by_home.get(structure.structure_id, [])
                birth_interval = max(1, self.config.home_birth_interval)
                can_birth_this_tick = self.current_tick % birth_interval == structure.structure_id % birth_interval
                if (
                    len(self.agents) < self.config.max_agents
                    and residents
                    and can_birth_this_tick
                    and structure.inventory.food >= self.config.reproduction_food_threshold
                    and structure.inventory.parts >= self.config.reproduction_parts_threshold
                ):
                    positions = self._adjacent_open_tiles(world_x, world_y)
                    if positions:
                        parent = self.rng.choice(residents)
                        genome = self.controller.mutate_genome(parent.genome, strength=0.05)
                        x, y = self.rng.choice(positions)
                        self._spawn_agent(
                            x=x,
                            y=y,
                            lineage_id=parent.lineage_id,
                            home_id=structure.structure_id,
                            genome=genome,
                            energy=self.config.offspring_energy,
                            generation=parent.generation + 1,
                        )
                        structure.inventory.remove(ResourceType.FOOD, self.config.reproduction_food_cost)
                        structure.inventory.remove(ResourceType.PARTS, self.config.reproduction_parts_cost)
                        births += 1

            if structure.health <= 0.0:
                world_x, world_y = self._world_position_of_structure(structure)
                if structure.inventory is not None:
                    for resource in ResourceType:
                        amount = structure.inventory.get(resource)
                        if amount > 0.0:
                            self.world.add_ground_resource(world_x, world_y, resource, amount)
                self.world.remove_structure(world_x, world_y)
                if structure.kind == StructureType.HOME:
                    for agent in list(self.agents.values()):
                        if agent.home_id == structure.structure_id:
                            agent.home_id = None
                deaths += 1
        return {"births": births, "deaths": deaths}

    def _collect_stats(
        self,
        births: int,
        deaths: int,
        extraction: dict[ResourceType, float],
        deliveries: dict[ResourceType, float],
        frontier_activated: int,
    ) -> SimulationStats:
        structure_counts = Counter()
        homes = 0
        road_length = 0
        storage_throughput = 0.0
        workshop_throughput = 0.0
        hub_scores: list[float] = []
        district_scores: list[float] = []
        for (world_x, world_y), structure in self.world.iter_active_structures():
            structure_counts[structure.kind.value] += 1
            if structure.kind == StructureType.HOME:
                homes += 1
            if structure.kind == StructureType.PATH:
                road_length += 1
            storage_throughput += structure.stored_throughput
            workshop_throughput += structure.processed_throughput
            if structure.kind in {StructureType.STORAGE, StructureType.WORKSHOP, StructureType.BEACON}:
                local_traffic = 0.0
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        chunk = self.world.get_chunk_for_tile(world_x + dx, world_y + dy, activate=False)
                        local_x, local_y = self.world.local_coords(world_x + dx, world_y + dy)
                        local_traffic += float(chunk.traffic[local_y, local_x])
                hub_scores.append(local_traffic)
            nearby_resources = self._district_resource_mix(world_x, world_y)
            if nearby_resources:
                total = sum(nearby_resources.values())
                dominant = max(nearby_resources.values())
                district_scores.append(dominant / total if total > 0.0 else 0.0)

        lineage_sizes = Counter(agent.lineage_id for agent in self.agents.values())
        stats = SimulationStats(
            population=len(self.agents),
            births=births,
            deaths=deaths,
            active_chunks=len(self.world.active_chunks),
            homes=homes,
            road_length=road_length,
            storage_throughput=storage_throughput,
            workshop_throughput=workshop_throughput,
            frontier_expansion_rate=float(frontier_activated),
            extraction_food=float(extraction[ResourceType.FOOD]),
            extraction_wood=float(extraction[ResourceType.WOOD]),
            extraction_stone=float(extraction[ResourceType.STONE]),
            delivered_food=float(deliveries[ResourceType.FOOD]),
            delivered_wood=float(deliveries[ResourceType.WOOD]),
            delivered_stone=float(deliveries[ResourceType.STONE]),
            delivered_parts=float(deliveries[ResourceType.PARTS]),
            district_specialization=float(np.mean(district_scores) if district_scores else 0.0),
            hub_centrality=float(np.mean(hub_scores) if hub_scores else 0.0),
            lineage_sizes=dict(lineage_sizes),
            structure_counts=dict(structure_counts),
        )
        return stats

    def _pull_workshop_inputs(self, world_x: int, world_y: int, inventory: Inventory) -> None:
        needed = {
            ResourceType.WOOD: max(0.0, self.config.workshop_wood_input - inventory.wood),
            ResourceType.STONE: max(0.0, self.config.workshop_stone_input - inventory.stone),
        }
        if needed[ResourceType.WOOD] <= 0.0 and needed[ResourceType.STONE] <= 0.0:
            return
        radius = max(1, self.config.workshop_resource_radius)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                self._pull_workshop_input_from_tile(world_x + dx, world_y + dy, inventory, needed)
                if needed[ResourceType.WOOD] <= 0.0 and needed[ResourceType.STONE] <= 0.0:
                    return

    def _pull_workshop_input_from_tile(
        self,
        world_x: int,
        world_y: int,
        inventory: Inventory,
        needed: dict[ResourceType, float],
    ) -> None:
        neighbor = self.world.get_structure(world_x, world_y)
        if neighbor is None or neighbor.inventory is None:
            return
        for resource, amount_needed in needed.items():
            if amount_needed <= 0.0:
                continue
            taken = neighbor.inventory.remove(resource, amount_needed)
            if taken > 0.0:
                inventory.add(resource, taken)
                needed[resource] -= taken

    def _opportunistic_deposit(self, agent: Agent, deliveries: dict[ResourceType, float]) -> None:
        if agent.carried_resource is None or agent.carried_amount <= 0.0:
            return
        carried = ResourceType(agent.carried_resource)
        target = self._opportunistic_deposit_target(agent, carried)
        if target is None or target.inventory is None:
            return
        amount = min(self.config.opportunistic_deposit_amount, agent.carried_amount)
        if amount <= 0.0:
            return
        target.inventory.add(carried, amount)
        target.stored_throughput += amount
        deliveries[carried] += amount
        agent.carried_amount -= amount
        if agent.carried_amount <= 0.0:
            agent.carried_amount = 0.0
            agent.carried_resource = None

    def _opportunistic_deposit_target(self, agent: Agent, resource: ResourceType):
        structures = self._accessible_inventory_structures(agent)
        if resource == ResourceType.FOOD:
            homes = [structure for structure in structures if structure.kind == StructureType.HOME]
            if homes:
                return min(homes, key=lambda structure: structure.inventory.food if structure.inventory else 0.0)
            storages = [structure for structure in structures if structure.kind == StructureType.STORAGE]
            if storages:
                return storages[0]
            return None
        if resource in {ResourceType.WOOD, ResourceType.STONE}:
            workshops = [structure for structure in structures if structure.kind == StructureType.WORKSHOP]
            if workshops:
                return workshops[0]
            storages = [structure for structure in structures if structure.kind == StructureType.STORAGE]
            if storages:
                return storages[0]
            return None
        homes = [structure for structure in structures if structure.kind == StructureType.HOME]
        if homes:
            return min(homes, key=lambda structure: structure.inventory.parts if structure.inventory else 0.0)
        storages = [structure for structure in structures if structure.kind == StructureType.STORAGE]
        return storages[0] if storages else None

    def _district_resource_mix(self, world_x: int, world_y: int) -> dict[str, float]:
        totals = defaultdict(float)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                terrain = self.world.terrain_at(world_x + dx, world_y + dy)
                if terrain in HARVESTABLE_TERRAIN_TO_RESOURCE:
                    resource = HARVESTABLE_TERRAIN_TO_RESOURCE[terrain]
                    totals[resource.value] += self.world.resource_amount_at(world_x + dx, world_y + dy)
        return totals

    def _choose_withdraw_resource(self, agent: Agent, inventory: Inventory) -> ResourceType | None:
        if agent.energy < 10.0 and inventory.food > 0.0:
            return ResourceType.FOOD
        if inventory.parts > 0.0:
            return ResourceType.PARTS
        if inventory.food > 0.0:
            return ResourceType.FOOD
        if inventory.wood > 0.0:
            return ResourceType.WOOD
        if inventory.stone > 0.0:
            return ResourceType.STONE
        return None

    def _accessible_inventory_structures(self, agent: Agent) -> list:
        structures = []
        for dx, dy in ((0, 0), *CARDINAL_OFFSETS):
            structure = self.world.get_structure(agent.x + dx, agent.y + dy)
            if structure is not None and structure.inventory is not None:
                structures.append(structure)
        return structures

    def _inventory_structure_for_direction(self, agent: Agent, direction: Direction):
        dx, dy = DIRECTION_VECTORS[direction]
        structure = self.world.get_structure(agent.x + dx, agent.y + dy)
        if structure is None or structure.inventory is None:
            return None
        return structure

    def _deposit_structure_for_direction(self, agent: Agent, direction: Direction):
        carried = None if agent.carried_resource is None else ResourceType(agent.carried_resource)
        if carried in {ResourceType.WOOD, ResourceType.STONE}:
            workshop = self._adjacent_inventory_of_kind(agent, StructureType.WORKSHOP)
            if workshop is not None:
                return workshop
        if carried == ResourceType.FOOD:
            home = self._adjacent_inventory_of_kind(agent, StructureType.HOME)
            if home is not None:
                return home
        return self._inventory_structure_for_direction(agent, direction)

    def _adjacent_inventory_of_kind(self, agent: Agent, kind: StructureType):
        for dx, dy in ((0, 0), *CARDINAL_OFFSETS):
            structure = self.world.get_structure(agent.x + dx, agent.y + dy)
            if structure is not None and structure.kind == kind and structure.inventory is not None:
                return structure
        return None

    def _home_structure_for_agent(self, agent: Agent):
        if agent.home_id is None:
            return None
        for _, structure in self.world.iter_active_structures():
            if structure.structure_id == agent.home_id:
                return structure
        return None

    def _cardinal_offsets(self) -> tuple[tuple[int, int], ...]:
        return CARDINAL_OFFSETS

    def _can_build_on(self, x: int, y: int) -> bool:
        if not self.world.is_passable(x, y):
            return False
        if self.world.get_structure(x, y) is not None:
            return False
        return self.world.terrain_at(x, y) != TerrainType.WATER

    def _can_build_path(self, x: int, y: int) -> bool:
        if not self._can_build_on(x, y):
            return False
        traffic = self._local_traffic(x, y, radius=1)
        adjacent_settlement = self._nearby_structure_count(
            x,
            y,
            kinds={StructureType.PATH, StructureType.HOME, StructureType.STORAGE, StructureType.WORKSHOP, StructureType.GATE},
            radius=1,
        )
        return traffic >= self.config.path_traffic_threshold or adjacent_settlement >= self.config.path_adjacent_settlement_threshold

    def _can_build_storage(self, x: int, y: int) -> bool:
        if not self._can_build_on(x, y):
            return False
        if self._nearest_structure_distance(x, y, StructureType.STORAGE) < self.config.storage_min_distance:
            return False
        traffic = self._local_traffic(x, y, radius=2)
        crossroads = self._nearby_structure_count(x, y, kinds={StructureType.PATH}, radius=1)
        access = self._access_score(x, y)
        return (
            traffic >= self.config.storage_hub_threshold
            or (crossroads >= self.config.storage_crossroads_threshold and access >= self.config.storage_access_threshold)
        )

    def _can_build_beacon(self, x: int, y: int) -> bool:
        if not self._can_build_on(x, y):
            return False
        if self._nearest_structure_distance(x, y, StructureType.BEACON) < self.config.beacon_min_distance:
            return False
        influence = self.world.influence_tiles.get((x, y), 0.0)
        frontierish = self.config.beacon_frontier_min <= influence <= self.config.influence_activation_threshold + 0.9
        return frontierish or self._local_traffic(x, y, radius=2) >= self.config.storage_hub_threshold * self.config.beacon_traffic_multiplier

    def _can_build_wall(self, x: int, y: int) -> bool:
        if not self._can_build_on(x, y):
            return False
        influence = self.world.influence_tiles.get((x, y), 0.0)
        if influence < self.config.wall_min_influence:
            return False
        nearby_hazard = max(
            self.world.hazard_at(x + dx, y + dy)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
        )
        exposure = sum(
            1
            for dx, dy in self._cardinal_offsets()
            if self.world.influence_tiles.get((x + dx, y + dy), 0.0) < self.config.wall_edge_influence
            or self.world.terrain_at(x + dx, y + dy) in {TerrainType.WATER, TerrainType.HAZARD}
        )
        return nearby_hazard >= self.config.wall_hazard_threshold or exposure >= self.config.wall_exposure_threshold

    def _can_build_gate(self, x: int, y: int) -> bool:
        if not self._can_build_on(x, y):
            return False
        wall_neighbors = 0
        path_neighbors = 0
        for dx, dy in self._cardinal_offsets():
            structure = self.world.get_structure(x + dx, y + dy)
            if structure is not None and structure.kind == StructureType.WALL:
                wall_neighbors += 1
            if structure is not None and structure.kind == StructureType.PATH:
                path_neighbors += 1
        return wall_neighbors >= self.config.gate_wall_neighbors_threshold and (
            path_neighbors >= self.config.gate_path_neighbors_threshold
            or self._local_traffic(x, y, radius=1) >= self.config.path_traffic_threshold
        )

    def _can_build_home(self, x: int, y: int) -> bool:
        if not self._can_build_on(x, y):
            return False
        if self.world.terrain_at(x, y) == TerrainType.HAZARD:
            return False
        influence = self.world.influence_tiles.get((x, y), 0.0)
        if influence < self.config.home_min_influence:
            return False
        if self._nearest_structure_distance(x, y, StructureType.HOME) < self.config.home_min_distance:
            return False
        local_surplus = self._local_resource_total(x, y, radius=self.config.home_resource_radius)
        access = self._access_score(x, y)
        competition = self._nearby_structure_count(x, y, kinds={StructureType.HOME}, radius=self.config.home_competition_radius)
        return (
            local_surplus >= self.config.home_surplus_threshold
            and access >= self.config.home_access_threshold
            and competition == 0
        )

    def _can_build_workshop(self, x: int, y: int) -> bool:
        if not self._can_build_on(x, y):
            return False
        if self._nearest_structure_distance(x, y, StructureType.WORKSHOP) < self.config.workshop_min_distance:
            return False
        raw_supply = self._local_resource_total(
            x,
            y,
            radius=self.config.workshop_resource_radius,
            terrains={TerrainType.FOREST, TerrainType.STONE},
        )
        access = self._access_score(x, y)
        return raw_supply >= self.config.workshop_resource_threshold and access >= self.config.workshop_access_threshold

    def _adjacent_open_tiles(self, x: int, y: int) -> list[tuple[int, int]]:
        positions = []
        occupied = set(self.agent_positions())
        radius = max(1, self.config.spawn_radius)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                tx = x + dx
                ty = y + dy
                if (tx, ty) in occupied:
                    continue
                if self.world.can_enter_tile(tx, ty) and self.world.get_structure(tx, ty) is None:
                    self.world.activate_tile(tx, ty)
                    positions.append((tx, ty))
        return positions

    def _local_traffic(self, x: int, y: int, radius: int) -> float:
        total = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                chunk = self.world.get_chunk_for_tile(x + dx, y + dy, activate=False)
                local_x, local_y = self.world.local_coords(x + dx, y + dy)
                total += float(chunk.traffic[local_y, local_x])
        return total

    def _local_resource_total(
        self,
        x: int,
        y: int,
        radius: int,
        terrains: set[TerrainType] | None = None,
    ) -> float:
        total = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                terrain = self.world.terrain_at(x + dx, y + dy)
                if terrains is not None and terrain not in terrains:
                    continue
                if terrain in HARVESTABLE_TERRAIN_TO_RESOURCE:
                    total += self.world.resource_amount_at(x + dx, y + dy) * self.world.resource_quality_at(x + dx, y + dy)
        return total

    def _nearest_structure_distance(self, x: int, y: int, kind: StructureType) -> int:
        best = 10**9
        for (world_x, world_y), structure in self.world.iter_active_structures():
            if structure.kind != kind:
                continue
            distance = abs(world_x - x) + abs(world_y - y)
            best = min(best, distance)
        return best

    def _nearby_structure_count(self, x: int, y: int, kinds: set[StructureType], radius: int) -> int:
        count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                structure = self.world.get_structure(x + dx, y + dy)
                if structure is not None and structure.kind in kinds:
                    count += 1
        return count

    def _access_score(self, x: int, y: int) -> float:
        score = 0.0
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                structure = self.world.get_structure(x + dx, y + dy)
                if structure is None:
                    continue
                distance = abs(dx) + abs(dy)
                weight = 1.0 / max(distance, 1)
                if structure.kind == StructureType.PATH:
                    score += 0.45 * weight
                elif structure.kind == StructureType.STORAGE:
                    score += 1.0 * weight
                elif structure.kind == StructureType.WORKSHOP:
                    score += 0.9 * weight
                elif structure.kind == StructureType.HOME:
                    score += 0.6 * weight
                elif structure.kind == StructureType.BEACON:
                    score += 0.7 * weight
        return score

    def _world_position_of_structure(self, structure) -> tuple[int, int]:
        for (chunk_x, chunk_y), chunk in self.world.active_chunks.items():
            if chunk.structures.get((structure.x, structure.y)) is structure:
                return chunk_x * self.config.chunk_size + structure.x, chunk_y * self.config.chunk_size + structure.y
        for (chunk_x, chunk_y), chunk in self.world.dormant_chunks.items():
            if chunk.structures.get((structure.x, structure.y)) is structure:
                return chunk_x * self.config.chunk_size + structure.x, chunk_y * self.config.chunk_size + structure.y
        raise ValueError("structure not found in world")
