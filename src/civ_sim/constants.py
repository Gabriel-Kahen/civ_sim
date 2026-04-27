from __future__ import annotations

from enum import IntEnum, StrEnum


class TerrainType(StrEnum):
    GRASS = "grass"
    FOREST = "forest"
    STONE = "stone"
    FERTILE = "fertile"
    WATER = "water"
    HAZARD = "hazard"


class StructureType(StrEnum):
    PATH = "path"
    STORAGE = "storage"
    BEACON = "beacon"
    WALL = "wall"
    GATE = "gate"
    HOME = "home"
    WORKSHOP = "workshop"


class ResourceType(StrEnum):
    FOOD = "food"
    WOOD = "wood"
    STONE = "stone"
    PARTS = "parts"


class ActionType(IntEnum):
    MOVE = 0
    HARVEST = 1
    PICKUP = 2
    DROP = 3
    DEPOSIT = 4
    WITHDRAW = 5
    BUILD_PATH = 6
    BUILD_STORAGE = 7
    BUILD_BEACON = 8
    BUILD_WALL = 9
    BUILD_GATE = 10
    BUILD_HOME = 11
    BUILD_WORKSHOP = 12
    REPAIR = 13
    IDLE = 14
    GUARD = 15


class Direction(IntEnum):
    CENTER = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


DIRECTION_VECTORS = {
    Direction.CENTER: (0, 0),
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}

CARDINAL_DIRECTIONS = (
    Direction.NORTH,
    Direction.EAST,
    Direction.SOUTH,
    Direction.WEST,
)

TERRAIN_ORDER = (
    TerrainType.GRASS,
    TerrainType.FOREST,
    TerrainType.STONE,
    TerrainType.FERTILE,
    TerrainType.WATER,
    TerrainType.HAZARD,
)

STRUCTURE_ORDER = (
    StructureType.PATH,
    StructureType.STORAGE,
    StructureType.BEACON,
    StructureType.WALL,
    StructureType.GATE,
    StructureType.HOME,
    StructureType.WORKSHOP,
)

RESOURCE_ORDER = (
    ResourceType.FOOD,
    ResourceType.WOOD,
    ResourceType.STONE,
    ResourceType.PARTS,
)

TERRAIN_INDEX = {terrain: index for index, terrain in enumerate(TERRAIN_ORDER)}
STRUCTURE_INDEX = {structure: index for index, structure in enumerate(STRUCTURE_ORDER)}
RESOURCE_INDEX = {resource: index for index, resource in enumerate(RESOURCE_ORDER)}
INDEX_TO_TERRAIN = {index: terrain for terrain, index in TERRAIN_INDEX.items()}
INDEX_TO_STRUCTURE = {index: structure for structure, index in STRUCTURE_INDEX.items()}

HARVESTABLE_TERRAIN_TO_RESOURCE = {
    TerrainType.FERTILE: ResourceType.FOOD,
    TerrainType.FOREST: ResourceType.WOOD,
    TerrainType.STONE: ResourceType.STONE,
}

PASSABLE_TERRAIN = {
    TerrainType.GRASS,
    TerrainType.FOREST,
    TerrainType.STONE,
    TerrainType.FERTILE,
    TerrainType.HAZARD,
}

BLOCKING_STRUCTURES = {
    StructureType.WALL,
    StructureType.STORAGE,
    StructureType.BEACON,
    StructureType.HOME,
    StructureType.WORKSHOP,
}

INVENTORY_STRUCTURES = {
    StructureType.HOME,
    StructureType.STORAGE,
    StructureType.WORKSHOP,
}
