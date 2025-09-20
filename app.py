# EV Persona Simulation Dash — Two-Day Simulation with Per-Day Plug/Drive Probability

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Heavy optional imports (Dash, plotly, pulp) are imported inside make_app
# so the module can be imported in test environments without these packages.


@dataclass
class EVAgent:
    id: int
    battery_kwh: float
    eff_kwh_per_mile: float
    daily_miles_mu: float
    daily_miles_sd: float
    starting_soc: float
    plugin_soc: float
    home_charge_prob: float
    drive_prob: float
    work_charge_prob: float
    home_charger_kw: float
    work_charger_kw: float
    arrive_home_mu: float
    arrive_home_sd: float
    depart_home_mu: float
    depart_home_sd: float
    target_soc: float
    archetype_name: str
    strategy: str

    def simulate_day(
        self, time_steps: int, dt_h: float, optimise: bool = False, n_days: int = 2
    ) -> Dict:
        """
        Simulate n_days of driving/charging behaviour.
        - Starts simulation at first plug-in event but does NOT rotate the day.
        - Still returns results aligned to real midnight-to-midnight for the last day.
        """
        steps_per_day = time_steps
        total_steps = steps_per_day * n_days

        energy = np.zeros(total_steps)
        plug_mask = np.zeros(total_steps, dtype=bool)
        consumption = np.zeros(total_steps)
        load_kw = np.zeros(total_steps)

        # Initial SoC before Day 1
        energy[0] = self.starting_soc * self.battery_kwh
        first_plug_idx = None

        for day in range(n_days):
            offset = day * steps_per_day
            depart_idx = int(
                np.clip(
                    np.random.normal(self.depart_home_mu, self.depart_home_sd) / dt_h,
                    0,
                    steps_per_day - 1,
                )
            )
            arrive_idx = int(
                np.clip(
                    np.random.normal(self.arrive_home_mu, self.arrive_home_sd) / dt_h,
                    0,
                    steps_per_day - 1,
                )
            )

            drives_today = np.random.random() <= self.drive_prob
            if drives_today:
                if depart_idx < arrive_idx:
                    drive_idx = np.arange(offset + depart_idx, offset + arrive_idx)
                else:
                    drive_idx = np.concatenate(
                        (
                            np.arange(offset + depart_idx, offset + steps_per_day),
                            np.arange(offset, offset + arrive_idx),
                        )
                    )
                miles_today = max(
                    0.0,
                    np.random.normal(
                        loc=self.daily_miles_mu,
                        scale=0.30 * max(1e-6, self.daily_miles_mu),
                    ),
                )
                energy_need = miles_today * self.eff_kwh_per_mile
                if drive_idx.size > 0 and energy_need > 0:
                    w = np.random.dirichlet(np.ones(drive_idx.size))
                    consumption[drive_idx] += energy_need * w

            planned_today = consumption[offset : offset + steps_per_day].sum()
            starting_energy = energy[offset]
            arrival_energy_est = max(0.0, starting_energy - planned_today)
            low_soc_threshold = self.plugin_soc * self.battery_kwh

            plug_today = False
            if arrival_energy_est <= low_soc_threshold:
                plug_today = True
            plug_today = np.random.random() <= self.home_charge_prob

            if plug_today:
                if arrive_idx <= depart_idx:
                    plug_mask[offset + arrive_idx : offset + depart_idx] = True
                else:
                    plug_mask[offset + arrive_idx : offset + steps_per_day] = True
                    plug_mask[offset : offset + depart_idx] = True

                if first_plug_idx is None:
                    first_plug_idx = offset + arrive_idx
                    # Pre-charge at first plug-in
                    energy[first_plug_idx] = max(
                        energy[first_plug_idx], self.target_soc * self.battery_kwh
                    )

        # Net energy loop
        for t in range(1, total_steps):
            e = max(0, energy[t - 1] - consumption[t])
            if plug_mask[t] and e < self.target_soc * self.battery_kwh:
                add = min(
                    self.home_charger_kw * dt_h, self.target_soc * self.battery_kwh - e
                )
                e += add
                load_kw[t] = add / dt_h
            energy[t] = e

        # --- Midnight-to-midnight alignment ---
        # Last full day always starts at a multiple of steps_per_day from index 0
        start_of_last_day = (n_days - 1) * steps_per_day
        last_day_slice = slice(start_of_last_day, start_of_last_day + steps_per_day)

        return {
            "soc": energy[last_day_slice] / self.battery_kwh,
            "load_kw": load_kw[last_day_slice],
            "plug_mask": plug_mask[last_day_slice],
        }


@dataclass
class Community:
    agents: List[EVAgent]

    def simulate(
        self, time_steps: int = 96, dt_h: float = 0.25, optimise: bool = False
    ):
        """Run a simulation and return aggregated results.

        Defaults provided so tests can call simulate() without arguments.
        """
        agg_soc = np.zeros(time_steps)
        agg_load = np.zeros(time_steps)
        agent_results = {}
        for agent in self.agents:
            result = agent.simulate_day(time_steps, dt_h, optimise=optimise)
            agent_results[agent.id] = result
            agg_soc += result["soc"]
            agg_load += result["load_kw"]
        avg_soc = agg_soc / len(self.agents) if len(self.agents) > 0 else agg_soc
        return {
            "time": np.arange(time_steps) * dt_h,
            "avg_soc": avg_soc,
            "agg_load": agg_load,
            "agents": agent_results,
            "agent_classes": self.agents,
        }


def build_community_from_archetypes(csv_path: str, n_agents: int = 50) -> Community:
    df = pd.read_csv(csv_path)
    df["prob"] = df["Percent of population"] / df["Percent of population"].sum()
    chosen = np.random.choice(df.index, size=n_agents, p=df["prob"])
    agents = []
    for i, idx in enumerate(chosen):
        row = df.loc[idx]
        miles_per_day = row["Miles per yr"] / 365.0
        eff_kwh_per_mile = 1.0 / row["Efficiency miperkWh"]

        # Compute starting SoC for infrequent chargers - will be somewhere between plug-in SoC and average UK driver SoC
        plug_in_soc = float(str(row["Plug-in SoC"]).strip("%")) / 100
        if row["Name"].lower().startswith("infrequent charging"):
            avg_uk_soc = 0.68
            soc = np.random.uniform(low=plug_in_soc, high=avg_uk_soc)
        else:
            soc = plug_in_soc

        arrive_hour = (
            pd.to_datetime(row["Plug-in time"]).hour
            + pd.to_datetime(row["Plug-in time"]).minute / 60
        )
        depart_hour = (
            pd.to_datetime(row["Plug-out time"]).hour
            + pd.to_datetime(row["Plug-out time"]).minute / 60
        )

        agents.append(
            EVAgent(
                id=i,
                battery_kwh=row["Battery kWh"],
                eff_kwh_per_mile=eff_kwh_per_mile,
                daily_miles_mu=miles_per_day,
                daily_miles_sd=miles_per_day * 0.2,
                starting_soc=soc,
                plugin_soc=plug_in_soc,
                home_charge_prob=row["Plug-in frequency per day"],
                drive_prob=row.get("Driving frequency per day", 1.0),
                work_charge_prob=0.0,
                home_charger_kw=row["Charger kW"],
                work_charger_kw=0.0,
                arrive_home_mu=arrive_hour,
                arrive_home_sd=0.5,
                depart_home_mu=depart_hour,
                depart_home_sd=0.5,
                target_soc=(
                    float(str(row["Target SoC"]).strip("%")) / 100
                    if "Target SoC" in row
                    else 1.0
                ),
                archetype_name=row["Name"],
                strategy="cheap",
            )
        )
    return Community(agents)


@dataclass
class Experiment:
    name: str
    hypothesis: str
    independent_vars: Dict[str, Any]
    control_vars: Dict[str, Any]
    dependent_vars: Dict[str, Any] = field(default_factory=dict)

    def run(self, community: Community, optimise: bool = False):
        res = community.simulate(
            time_steps=self.control_vars.get("time_steps", 96),
            dt_h=self.control_vars.get("dt_h", 0.25),
            optimise=optimise,
        )
        self.dependent_vars = res
        return res


def make_app(csv_path="archetypes.csv", optimise=False, n_agents=50):
    from dash import Dash, dcc, html, Input, Output
    import plotly.graph_objects as go

    community = build_community_from_archetypes(csv_path, n_agents=n_agents)
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H3("Community V2X Simulation — Per-Day Plug and Drive Probabilities"),
            dcc.Slider(
                id="n_agents",
                min=5,
                max=n_agents,
                step=5,
                value=n_agents,
                tooltip={"always_visible": True},
            ),
            dcc.Graph(id="soc_graph"),
            dcc.Dropdown(id="agent_selector", placeholder="Select agent to view"),
            dcc.Graph(id="agent_soc_graph"),
            dcc.Graph(id="load_graph"),
        ]
    )

    @app.callback(
        [
            Output("soc_graph", "figure"),
            Output("load_graph", "figure"),
            Output("agent_selector", "options"),
            Output("agent_selector", "value"),
        ],
        Input("n_agents", "value"),
    )
    def update_graphs(n_agents):
        comm = build_community_from_archetypes(csv_path, n_agents=n_agents)
        exp = Experiment(
            name="Per-Day Plug/Drive Simulation",
            hypothesis="Charging and driving sampled per day",
            independent_vars={"n_agents": n_agents},
            control_vars={"time_steps": 96, "dt_h": 0.25},
        )
        res = exp.run(comm, optimise=optimise)
        t = res["time"]

        # --- Compute percentiles across agents for SoC ---
        soc_matrix = np.vstack([ares["soc"] for ares in res["agents"].values()])
        mean_soc = soc_matrix.mean(axis=0) * 100
        p2p5 = np.percentile(soc_matrix * 100, 2.5, axis=0)
        p97p5 = np.percentile(soc_matrix * 100, 97.5, axis=0)

        # --- Compute % plugged in ---
        plug_matrix = np.vstack([ares["plug_mask"] for ares in res["agents"].values()])
        pct_plugged = plug_matrix.mean(axis=0) * 100

        # --- Build figure with shaded percentiles + bars ---
        soc_fig = go.Figure()

        # Add shaded confidence interval (fill between percentiles)
        soc_fig.add_trace(
            go.Scatter(
                x=t,
                y=p97p5,
                line=dict(width=0),
                mode="lines",
                name="97.5th percentile",
                showlegend=False,
            )
        )
        soc_fig.add_trace(
            go.Scatter(
                x=t,
                y=p2p5,
                line=dict(width=0),
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0, 100, 250, 0.2)",
                name="2.5–97.5 percentile range",
            )
        )

        # Add mean SoC line
        soc_fig.add_trace(
            go.Scatter(
                x=t,
                y=mean_soc,
                mode="lines",
                name="Mean SoC",
                line=dict(color="blue", width=2),
            )
        )

        # Add bars for % plugged in (secondary y-axis)
        soc_fig.add_trace(
            go.Bar(
                x=t,
                y=pct_plugged,
                name="% Plugged In",
                marker_color="rgba(0, 200, 0, 0.4)",
                yaxis="y2",
            )
        )

        # Layout with dual y-axis: left = SoC, right = % plugged in
        soc_fig.update_layout(
            title="Average State of Charge (%) with 95% Range & Plugged-in %",
            xaxis_title="Hour",
            yaxis=dict(title="SoC %", range=[0, 100]),
            yaxis2=dict(
                title="% Plugged In",
                overlaying="y",
                side="right",
                range=[0, 100],
                showgrid=False,
            ),
            barmode="overlay",
            bargap=0,
        )

        # Load graph (kW, not percentage)
        load_fig = go.Figure(go.Scatter(x=t, y=res["agg_load"], mode="lines"))
        load_fig.update_layout(
            title="Aggregate Demand (kW)", xaxis_title="Hour", yaxis_title="kW"
        )

        options = [
            {"label": f"Agent {aid} — {comm.agents[aid].archetype_name}", "value": aid}
            for aid in range(len(comm.agents))
        ]
        first_agent = options[0]["value"] if options else None
        return soc_fig, load_fig, options, first_agent

    @app.callback(
        Output("agent_soc_graph", "figure"),
        [Input("agent_selector", "value"), Input("n_agents", "value")],
    )
    def update_agent_chart(agent_id, n_agents):
        comm = build_community_from_archetypes(csv_path, n_agents=n_agents)
        res = comm.simulate(time_steps=96, dt_h=0.25, optimise=optimise)
        if agent_id not in res["agents"]:
            return go.Figure()
        ares = res["agents"][agent_id]
        t = res["time"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=ares["soc"] * 100, mode="lines", name="SoC %"))
        fig.add_trace(
            go.Scatter(
                x=t,
                y=[100 if m else 0 for m in ares["plug_mask"]],
                fill="tozeroy",
                mode="none",
                fillcolor="rgba(0,200,0,0.2)",
                name="Plugged in",
            )
        )
        fig.update_layout(
            title=f"Agent {agent_id} ({comm.agents[agent_id].archetype_name}) SoC Profile",
            xaxis_title="Hour",
            yaxis_title="SoC %",
            yaxis=dict(range=[0, 100]),
        )
        return fig

    return app


if __name__ == "__main__":
    app = make_app(csv_path="archetypes.csv", optimise=False, n_agents=100)
    app.run(host="0.0.0.0", port=8050, debug=True)
