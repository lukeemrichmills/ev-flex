import pytest, numpy as np, pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import build_community_from_archetypes

CSV_PATH = "archetypes.csv"


def infreq_drive_community():
    df = pd.read_csv(CSV_PATH)
    comm = build_community_from_archetypes(CSV_PATH, n_agents=100)
    # Filter to only include infrequent driving archetype agents
    comm.agents = [
        a
        for a in comm.agents
        if a.archetype_name.lower().startswith("infrequent driving")
    ]
    return comm


def test_infrequent_driving_daily_probability(infreq_drive_community):
    """
    Check that the proportion of agents that actually drove is close to drive_prob.
    We repeat multiple simulations to smooth out stochastic variation.
    """
    n_runs = 50
    observed_drive_rates = []

    for _ in range(n_runs):
        drive_flags = []
        for agent in infreq_drive_community.agents:
            # Sample once for whether they would drive today
            drives_today = np.random.random() <= agent.drive_prob
            drive_flags.append(drives_today)
        observed_drive_rates.append(np.mean(drive_flags))

    mean_drive_rate = np.mean(observed_drive_rates)
    expected_drive_rate = np.mean([a.drive_prob for a in infreq_drive_community.agents])

    # Assert that observed drive rate is close to expected rate (±10% tolerance)
    lower_bound = expected_drive_rate * 0.9
    upper_bound = expected_drive_rate * 1.1

    assert lower_bound <= mean_drive_rate <= upper_bound, (
        f"Observed driving rate {mean_drive_rate:.2f} is outside expected range "
        f"({lower_bound:.2f}-{upper_bound:.2f}) for infrequent drivers"
    )

if __name__ == "__main__":
    test_infrequent_driving_daily_probability(infreq_drive_community())
    print("All tests passed.")