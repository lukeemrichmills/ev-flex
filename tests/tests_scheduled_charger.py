import pytest, numpy as np, pandas as pd
import os, sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import build_community_from_archetypes

CSV_PATH = "archetypes.csv"


def scheduled_comm():
    df = pd.read_csv(CSV_PATH)
    comm = build_community_from_archetypes(CSV_PATH, n_agents=50)
    comm.agents = [
        a
        for a in comm.agents
        if a.archetype_name.lower().startswith("scheduled charging")
    ]
    return comm, df[df["Name"].str.lower().str.startswith("scheduled charging")]


def time_to_index(time_str, dt_h=0.25):
    """Convert time string to quarter-hour index, supports 12h or 24h format."""
    parsed = pd.to_datetime(time_str, format=None)
    hours = parsed.hour + parsed.minute / 60
    return int(hours / dt_h)


def test_scheduled_charging_window_alignment(scheduled_comm):
    """
    Ensure that plug_mask for scheduled charging agents is True during the scheduled window
    (from Plug-in time to Plug-out time) and False outside that window.
    """
    comm, archetype_df = scheduled_comm
    assert len(comm.agents) > 0, "No scheduled charging agents found to test."

    # Only test a small number to keep test quick
    agents_to_check = comm.agents[:5]

    for agent in agents_to_check:
        row = archetype_df.iloc[0]  # assume consistent plug times for this archetype
        plug_in_idx = time_to_index(row["Plug-in time"])
        plug_out_idx = time_to_index(row["Plug-out time"])

        res = agent.simulate_day(time_steps=96, dt_h=0.25, n_days=2)
        mask = res["plug_mask"]

        # Determine which indices are expected to be True
        if plug_in_idx <= plug_out_idx:
            expected_indices = np.arange(plug_in_idx, plug_out_idx)
        else:
            expected_indices = np.concatenate(
                (np.arange(plug_in_idx, 96), np.arange(0, plug_out_idx))
            )

        # Assert that majority of expected_indices are plugged in
        plugged_fraction = np.mean(mask[expected_indices])
        assert plugged_fraction > 0.8, (
            f"Agent {agent.id} did not stay plugged in for most of the scheduled window "
            f"(plugged fraction={plugged_fraction:.2f})."
        )

        # Assert that majority of outside indices are unplugged
        outside_indices = np.setdiff1d(np.arange(96), expected_indices)
        unplugged_fraction = np.mean(~mask[outside_indices])
        assert unplugged_fraction > 0.8, (
            f"Agent {agent.id} was plugged in too often outside the scheduled window "
            f"(unplugged fraction={unplugged_fraction:.2f})."
        )


if __name__ == "__main__":
    test_scheduled_charging_window_alignment(scheduled_comm())
    print("All tests passed.")
