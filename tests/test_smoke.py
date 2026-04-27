from pathlib import Path

from civ_sim.io import load_simulation, save_simulation
from civ_sim.config import SimConfig
from civ_sim.sim import Simulation


def test_short_simulation_run():
    sim = Simulation.create(SimConfig(initial_agents=8, max_agents=32, initial_active_radius_chunks=1))
    for _ in range(5):
        stats = sim.step()
    assert stats.population >= 0
    assert stats.active_chunks > 0


def test_save_load_round_trip(tmp_path: Path):
    sim = Simulation.create(SimConfig(initial_agents=8, max_agents=32, initial_active_radius_chunks=1))
    for _ in range(3):
        sim.step()
    path = tmp_path / "state.json"
    save_simulation(sim, path)
    loaded = load_simulation(path)
    assert loaded.current_tick == sim.current_tick
    assert len(loaded.agents) == len(sim.agents)
