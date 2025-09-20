import pytest, numpy as np, pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import build_community_from_archetypes, Community

CSV_PATH = "archetypes.csv"


# @pytest.fixture
def avg_community():
    df = pd.read_csv(CSV_PATH)
    avg_names = df[df["Name"].str.contains("Average", case=False)]["Name"].values
    comm = build_community_from_archetypes(CSV_PATH, n_agents=50)
    comm.agents = [a for a in comm.agents if a.archetype_name in avg_names]
    return comm


def test_avg_soc_range(avg_community):
    res = avg_community.simulate()
    start, end = res["avg_soc"][0], res["avg_soc"][-1]
    assert 0.2 < start < 0.9, "Starting SoC unrealistic for average UK driver"


if __name__ == "__main__":
    test_avg_soc_range(avg_community())
    print("All tests passed.")
