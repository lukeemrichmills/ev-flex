import pytest, numpy as np, pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import build_community_from_archetypes, Community

CSV_PATH = "archetypes.csv"


def infreq_charge_community():
    df = pd.read_csv(CSV_PATH)
    comm = build_community_from_archetypes(CSV_PATH, n_agents=1000)
    comm.agents = [a for a in comm.agents if a.archetype_name == "Infrequent charging"]
    return comm


def test_infrequent_charging_day_frequency(infreq_charge_community):
    """Check proportion of days with any charging events for infrequent chargers."""
    n_sims = 10  # repeat simulation multiple times to smooth randomness
    daily_charging_counts = []
    for _ in range(n_sims):
        res = infreq_charge_community.simulate(time_steps=96, dt_h=0.25)
        # For each agent, check if there was ANY charging at all that day
        charging_days = [np.any(ares["plug_mask"]) for ares in res["agents"].values()]
        daily_charging_counts.extend(charging_days)

    charging_rate = np.mean(daily_charging_counts)
    # Expect infrequent chargers to charge <40% of days on average
    assert (
        charging_rate < 0.4
    ), f"Infrequent chargers charged too often: {charging_rate:.2f}"
    # Also ensure they're not never charging
    assert (
        charging_rate > 0.05
    ), f"Infrequent chargers almost never charged: {charging_rate:.2f}"


if __name__ == "__main__":
    test_infrequent_charging_day_frequency(infreq_charge_community())
    print("All tests passed.")
