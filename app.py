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


def make_app(
    csv_path="archetypes.csv",
    csv_path_h1="archetypes_h1.csv",
    optimise=False,
    n_agents=50,
):
    from dash import Dash, dcc, html, Input, Output, State, dash_table
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    import random

    df_base = pd.read_csv(csv_path)
    df_base_h1 = pd.read_csv(csv_path_h1)

    # ---------- Helper function for SoC figure ----------
    def build_soc_fig(res, title):
        t = np.array(res["time"])
        soc_matrix = np.array(res["soc_matrix"])
        mean_soc = soc_matrix.mean(axis=0)
        p2p5 = np.percentile(soc_matrix, 2.5, axis=0)
        p97p5 = np.percentile(soc_matrix, 97.5, axis=0)
        pct_plugged = np.array(res["pct_plugged"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=p97p5, line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=t,
                y=p2p5,
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0,100,250,0.2)",
                name="95% range",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=mean_soc,
                mode="lines",
                name="Mean SoC",
                line=dict(color="blue", width=2),
            )
        )
        fig.add_trace(
            go.Bar(
                x=t,
                y=pct_plugged,
                name="% Charging",
                marker_color="rgba(0,200,0,0.4)",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Hour",
            yaxis=dict(title="SoC %", range=[0, 100]),
            yaxis2=dict(
                title="% Charging", overlaying="y", side="right", range=[0, 100]
            ),
            barmode="overlay",
            bargap=0,
        )
        return fig

    app = Dash(__name__)

    # ---- Layout ----
    app.layout = html.Div(
        [
            # Stores to cache results
            dcc.Store(id="res1_store"),
            dcc.Store(id="res2_store"),
            html.H2("1️⃣ Experiment Setup"),
            html.P(
                "Compare and modify archetype inputs for Experiment 2 below. Adjust population shares, plug-in times, and frequencies to explore different scenarios."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Experiment 1: Original Archetypes"),
                            dash_table.DataTable(
                                id="archetype_table_exp1",
                                columns=[{"name": c, "id": c} for c in df_base.columns],
                                data=df_base.to_dict("records"),
                                editable=False,
                                style_table={"overflowX": "auto"},
                                style_cell={"minWidth": "80px", "textAlign": "center"},
                            ),
                        ],
                        style={
                            "width": "45%",
                            "display": "inline-block",
                            "backgroundColor": "#f5f8ff",
                            "padding": "10px",
                            "borderRadius": "8px",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "inline-block",
                            "width": "2%",
                            "backgroundColor": "#ccc",
                            "margin": "0 1%",
                            "borderRadius": "4px",
                        }
                    ),
                    html.Div(
                        [
                            html.H4("Experiment 2: Editable Archetypes"),
                            dash_table.DataTable(
                                id="archetype_table_exp2",
                                columns=[
                                    {"name": "Name", "id": "Name", "editable": False},
                                    {
                                        "name": "Percent of population",
                                        "id": "Percent of population",
                                        "type": "numeric",
                                        "editable": True,
                                    },
                                    {
                                        "name": "Plug-in frequency per day",
                                        "id": "Plug-in frequency per day",
                                        "type": "numeric",
                                        "editable": True,
                                    },
                                    {
                                        "name": "Plug-in time",
                                        "id": "Plug-in time",
                                        "editable": True,
                                    },
                                    {
                                        "name": "Plug-out time",
                                        "id": "Plug-out time",
                                        "editable": True,
                                    },
                                    {
                                        "name": "Driving frequency per day",
                                        "id": "Driving frequency per day",
                                        "type": "numeric",
                                        "editable": True,
                                    },
                                ],
                                data=df_base_h1.to_dict("records"),
                                editable=True,
                                style_table={"overflowX": "auto"},
                                style_cell={"minWidth": "80px", "textAlign": "center"},
                            ),
                        ],
                        style={
                            "width": "45%",
                            "display": "inline-block",
                            "backgroundColor": "#f8fff5",
                            "padding": "10px",
                            "borderRadius": "8px",
                        },
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Hr(),
            html.H2("2️⃣ Aggregate Simulation Results"),
            dcc.Slider(
                id="n_agents",
                min=5,
                max=n_agents,
                step=50,
                value=100,
                tooltip={"always_visible": True},
            ),
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                style={"position": "relative", "top": "-900px"},
                children=html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id="soc_graph_exp1")],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "paddingRight": "1%",
                                    },
                                ),
                                html.Div(
                                    [dcc.Graph(id="soc_graph_exp2")],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "paddingLeft": "1%",
                                    },
                                ),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                        html.Div(
                            [dcc.Graph(id="agg_demand_graph")],
                            style={"marginBottom": "30px"},
                        ),
                        html.Hr(),
                        html.H2("3️⃣ Key Performance Indicators"),
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id="peak_demand_chart")],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "paddingRight": "1%",
                                    },
                                ),
                                html.Div(
                                    [dcc.Graph(id="total_demand_chart")],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "paddingLeft": "1%",
                                    },
                                ),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                        html.Hr(),
                        html.H2("4️⃣ Individual Agent Exploration"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Experiment 1 Agent"),
                                        dcc.Dropdown(
                                            id="agent_selector_exp1",
                                            placeholder="Select agent (Exp 1)",
                                        ),
                                    ],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "paddingRight": "1%",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label("Experiment 2 Agent"),
                                        dcc.Dropdown(
                                            id="agent_selector_exp2",
                                            placeholder="Select agent (Exp 2)",
                                        ),
                                    ],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "paddingLeft": "1%",
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id="agent_soc_exp1")],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "paddingRight": "1%",
                                    },
                                ),
                                html.Div(
                                    [dcc.Graph(id="agent_soc_exp2")],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "paddingLeft": "1%",
                                    },
                                ),
                            ]
                        ),
                    ]
                ),
            ),
        ]
    )

    # ---- Callbacks ----
    from dash import ctx

    @app.callback(
        [
            Output("soc_graph_exp1", "figure"),
            Output("soc_graph_exp2", "figure"),
            Output("agg_demand_graph", "figure"),
            Output("peak_demand_chart", "figure"),
            Output("total_demand_chart", "figure"),
            Output("agent_selector_exp1", "options"),
            Output("agent_selector_exp2", "options"),
            Output("agent_selector_exp1", "value"),
            Output("agent_selector_exp2", "value"),
            Output("res1_store", "data"),
            Output("res2_store", "data"),
        ],
        [Input("n_agents", "value"), Input("archetype_table_exp2", "data")],
    )
    def run_and_render(n_agents_val, table_data_exp2):
        # Import your build_community_from_archetypes & Experiment here
        comm1 = build_community_from_archetypes(csv_path, n_agents=n_agents_val)
        res1_raw = Experiment(
            "Experiment 1", "Baseline", {}, {"time_steps": 96, "dt_h": 0.25}
        ).run(comm1, optimise=optimise)

        # Convert to JSON-safe dict
        res1 = {
            "time": [float(x) for x in res1_raw["time"]],
            "agg_load": [float(x) for x in res1_raw["agg_load"]],
            "agents": {
                str(k): {
                    "soc": list(map(float, v["soc"])),
                    "plug_mask": list(map(bool, v["plug_mask"])),
                }
                for k, v in res1_raw["agents"].items()
            },
            "soc_matrix": np.vstack(
                [v["soc"] * 100 for v in res1_raw["agents"].values()]
            ).tolist(),
            "pct_plugged": (
                np.vstack([v["plug_mask"] for v in res1_raw["agents"].values()]).mean(
                    axis=0
                )
                * 100
            ).tolist(),
        }

        df_mod = pd.DataFrame(table_data_exp2)
        if df_mod["Percent of population"].sum() > 0:
            df_mod["Percent of population"] = (
                df_mod["Percent of population"] / df_mod["Percent of population"].sum()
            ) * 100
        temp_csv = "/tmp/experiment2.csv"
        df_mod.to_csv(temp_csv, index=False)

        comm2 = build_community_from_archetypes(temp_csv, n_agents=n_agents_val)
        res2_raw = Experiment(
            "Experiment 2", "User modified", {}, {"time_steps": 96, "dt_h": 0.25}
        ).run(comm2, optimise=optimise)

        res2 = {
            "time": [float(x) for x in res2_raw["time"]],
            "agg_load": [float(x) for x in res2_raw["agg_load"]],
            "agents": {
                str(k): {
                    "soc": list(map(float, v["soc"])),
                    "plug_mask": list(map(bool, v["plug_mask"])),
                }
                for k, v in res2_raw["agents"].items()
            },
            "soc_matrix": np.vstack(
                [v["soc"] * 100 for v in res2_raw["agents"].values()]
            ).tolist(),
            "pct_plugged": (
                np.vstack([v["plug_mask"] for v in res2_raw["agents"].values()]).mean(
                    axis=0
                )
                * 100
            ).tolist(),
        }

        soc_fig1 = build_soc_fig(res1, "Experiment 1: Avg SoC")
        soc_fig2 = build_soc_fig(res2, "Experiment 2: Avg SoC")
        agg_fig = go.Figure()
        agg_fig.add_trace(go.Scatter(x=res1["time"], y=res1["agg_load"], name="Exp 1"))
        agg_fig.add_trace(go.Scatter(x=res2["time"], y=res2["agg_load"], name="Exp 2"))
        agg_fig.update_layout(
            title="Aggregate Community Demand (kW)",
            xaxis_title="Hour",
            yaxis_title="kW",
        )

        peak_fig = go.Figure(
            [
                go.Bar(name="Exp 1", x=["Peak Demand"], y=[max(res1["agg_load"])]),
                go.Bar(name="Exp 2", x=["Peak Demand"], y=[max(res2["agg_load"])]),
            ]
        )
        total_fig = go.Figure(
            [
                go.Bar(
                    name="Exp 1", x=["Total Demand"], y=[sum(res1["agg_load"]) * 0.25]
                ),
                go.Bar(
                    name="Exp 2", x=["Total Demand"], y=[sum(res2["agg_load"]) * 0.25]
                ),
            ]
        )

        sample_ids_1 = random.sample(
            range(len(comm1.agents)), min(10, len(comm1.agents))
        )
        sample_ids_2 = random.sample(
            range(len(comm2.agents)), min(10, len(comm2.agents))
        )
        options1 = [
            {"label": f"Agent {i} — {comm1.agents[i].archetype_name}", "value": i}
            for i in sample_ids_1
        ]
        options2 = [
            {"label": f"Agent {i} — {comm2.agents[i].archetype_name}", "value": i}
            for i in sample_ids_2
        ]

        default_val_1 = sample_ids_1[0] if sample_ids_1 else None
        default_val_2 = sample_ids_2[0] if sample_ids_2 else None

        return (
            soc_fig1,
            soc_fig2,
            agg_fig,
            peak_fig,
            total_fig,
            options1,
            options2,
            default_val_1,
            default_val_2,
            res1,
            res2,
        )

    @app.callback(
        [Output("agent_soc_exp1", "figure"), Output("agent_soc_exp2", "figure")],
        [
            Input("agent_selector_exp1", "value"),
            Input("agent_selector_exp2", "value"),
            Input("res1_store", "data"),
            Input("res2_store", "data"),
        ],
    )
    def render_agents(agent_id1, agent_id2, res1, res2):
        def plot_agent(res, agent_id, title):
            if not res or agent_id is None:
                return go.Figure()
            a = res["agents"].get(str(agent_id))
            if a is None:
                return go.Figure()
            t = res["time"]
            soc = [x * 100 if max(a["soc"]) <= 1 else x for x in a["soc"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=soc, name="SoC %"))
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=[100 if p else 0 for p in a["plug_mask"]],
                    fill="tozeroy",
                    mode="none",
                    fillcolor="rgba(0,200,0,0.2)",
                )
            )
            fig.update_layout(title=title, yaxis=dict(range=[0, 100]))
            return fig

        return plot_agent(
            res1, agent_id1, f"Experiment 1: Agent {agent_id1}"
        ), plot_agent(res2, agent_id2, f"Experiment 2: Agent {agent_id2}")

    return app


if __name__ == "__main__":
    app = make_app(
        csv_path="archetypes.csv",
        optimise=False,
        n_agents=1000,  # per experiment
    )
    app.run(host="0.0.0.0", port=8050, debug=True)
