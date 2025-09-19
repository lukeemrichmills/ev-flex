# EV Persona Simulation Dash — Cyclic-Day Simulation (Departure-Aligned)

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pulp

@dataclass
class EVAgent:
    id: int
    battery_kwh: float
    eff_kwh_per_km: float
    daily_km_mu: float
    daily_km_sd: float
    soc: float
    home_charge_prob: float
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

    def simulate_day(self, time_steps: int, dt_h: float, optimise: bool = False) -> Dict:
        """Simulate a day starting at departure time (cyclic-day approach) and rotate results back to 0–24h."""

        # 1) Sample arrival and departure indices (clock time)
        depart_idx = int(np.clip(np.random.normal(self.depart_home_mu, self.depart_home_sd) / dt_h, 0, time_steps - 1))
        arrive_idx = int(np.clip(np.random.normal(self.arrive_home_mu, self.arrive_home_sd) / dt_h, 0, time_steps - 1))

        # Build plug-in mask (True when plugged in at home)
        plug_mask = np.zeros(time_steps, dtype=bool)
        if np.random.random() <= self.home_charge_prob:
            if arrive_idx <= depart_idx:
                plug_mask[arrive_idx:depart_idx] = True
            else:
                plug_mask[arrive_idx:] = True
                plug_mask[:depart_idx] = True

        # Build driving mask (True when vehicle is away/unplugged)
        drive_mask = np.zeros(time_steps, dtype=bool)
        if depart_idx < arrive_idx:
            drive_mask[depart_idx:arrive_idx] = True
        else:
            drive_mask[depart_idx:] = True
            drive_mask[:arrive_idx] = True

        # 2) Rotate masks so t=0 = departure time
        shift = depart_idx
        plug_mask = np.roll(plug_mask, -shift)
        drive_mask = np.roll(drive_mask, -shift)

        # 3) Initialise energy array, warm-start to target if plugged overnight
        energy = np.zeros(time_steps)
        energy[0] = self.soc * self.battery_kwh
        if plug_mask[0]:
            energy[0] = max(energy[0], self.target_soc * self.battery_kwh)

        # 4) Sample total daily energy need and distribute over driving interval
        daily_energy_mu = self.daily_km_mu * self.eff_kwh_per_km
        daily_energy_sd = daily_energy_mu * 0.3
        energy_need = max(0, np.random.normal(daily_energy_mu, daily_energy_sd))
        drive_idxs = np.where(drive_mask)[0]
        consumption_schedule = np.zeros(time_steps)
        if drive_idxs.size > 0:
            weights = np.random.dirichlet(np.ones(drive_idxs.size))
            consumption_schedule[drive_idxs] = energy_need * weights

        # 5) Step through the day applying driving depletion and charging
        load_kw = np.zeros(time_steps)
        for t in range(1, time_steps):
            # Deduct driving energy
            energy[t] = max(0, energy[t-1] - consumption_schedule[t])

            # Apply charging if plugged and below target SoC
            if plug_mask[t] and energy[t] < self.target_soc * self.battery_kwh:
                add = min(self.home_charger_kw * dt_h, self.target_soc * self.battery_kwh - energy[t])
                energy[t] += add
                load_kw[t] = add / dt_h

        # 6) Rotate results back to clock time for plotting
        energy = np.roll(energy, shift)
        plug_mask = np.roll(plug_mask, shift)
        load_kw = np.roll(load_kw, shift)

        return {"soc": energy / self.battery_kwh, "load_kw": load_kw, "plug_mask": plug_mask}

@dataclass
class Community:
    agents: List[EVAgent]

    def simulate(self, time_steps: int, dt_h: float, optimise: bool = False):
        agg_soc = np.zeros(time_steps)
        agg_load = np.zeros(time_steps)
        agent_results = {}
        for agent in self.agents:
            result = agent.simulate_day(time_steps, dt_h, optimise=optimise)
            agent_results[agent.id] = result
            agg_soc += result["soc"]
            agg_load += result["load_kw"]
        return {"time": np.arange(time_steps) * dt_h, "avg_soc": agg_soc / len(self.agents), "agg_load": agg_load, "agents": agent_results}

def build_community_from_archetypes(csv_path: str, n_agents: int = 50) -> Community:
    df = pd.read_csv(csv_path)
    df['prob'] = df['Percent of population'] / df['Percent of population'].sum()
    chosen = np.random.choice(df.index, size=n_agents, p=df['prob'])
    agents = []
    for i, idx in enumerate(chosen):
        row = df.loc[idx]
        km_per_day = (row['Miles per yr'] / 365.0) * 1.609
        eff_kwh_per_km = 1.0 / (row['Efficiency miperkWh'] * 1.609)
        arrive_hour = pd.to_datetime(row['Plug-in time']).hour + pd.to_datetime(row['Plug-in time']).minute / 60
        depart_hour = pd.to_datetime(row['Plug-out time']).hour + pd.to_datetime(row['Plug-out time']).minute / 60
        agents.append(EVAgent(id=i, battery_kwh=row['Battery kWh'], eff_kwh_per_km=eff_kwh_per_km,
                              daily_km_mu=km_per_day, daily_km_sd=km_per_day * 0.2,
                              soc=float(str(row['Plug-in SoC']).strip('%')) / 100,
                              home_charge_prob=row['Plug-in frequency per day'], work_charge_prob=0.0,
                              home_charger_kw=row['Charger kW'], work_charger_kw=0.0,
                              arrive_home_mu=arrive_hour, arrive_home_sd=0.5,
                              depart_home_mu=depart_hour, depart_home_sd=0.5,
                              target_soc=float(str(row['Target SoC']).strip('%')) / 100 if 'Target SoC' in row else 1.0,
                              archetype_name=row['Name'], strategy="cheap"))
    return Community(agents)

@dataclass
class Experiment:
    name: str
    hypothesis: str
    independent_vars: Dict[str, Any]
    control_vars: Dict[str, Any]
    dependent_vars: Dict[str, Any] = field(default_factory=dict)

    def run(self, community: Community, optimise: bool = False):
        res = community.simulate(time_steps=self.control_vars.get("time_steps", 96), dt_h=self.control_vars.get("dt_h", 0.25), optimise=optimise)
        self.dependent_vars = res
        return res

def make_app(csv_path="archetypes.csv", optimise=False):
    community = build_community_from_archetypes(csv_path, n_agents=20)
    app = Dash(__name__)
    app.layout = html.Div([
        html.H3("Community V2X Simulation — Cyclic-Day Approach"),
        dcc.Slider(id="n_agents", min=5, max=50, step=5, value=20, tooltip={"always_visible": True}),
        dcc.Graph(id="soc_graph"),
        dcc.Graph(id="load_graph"),
        dcc.Dropdown(id="agent_selector", placeholder="Select agent to view"),
        dcc.Graph(id="agent_soc_graph"),
    ])

    @app.callback(
        [Output("soc_graph", "figure"), Output("load_graph", "figure"), Output("agent_selector", "options"), Output("agent_selector", "value")],
        Input("n_agents", "value")
    )
    def update_graphs(n_agents):
        comm = build_community_from_archetypes(csv_path, n_agents=n_agents)
        exp = Experiment(name="Cyclic-Day Simulation", hypothesis="SoC starts at post-overnight value and declines during drive interval", independent_vars={"n_agents": n_agents}, control_vars={"time_steps": 96, "dt_h": 0.25})
        res = exp.run(comm, optimise=optimise)
        t = res["time"]
        soc_fig = go.Figure(go.Scatter(x=t, y=res["avg_soc"]*100, mode="lines"))
        load_fig = go.Figure(go.Scatter(x=t, y=res["agg_load"], mode="lines"))
        options = [{"label": f"Agent {aid} — {comm.agents[aid].archetype_name}", "value": aid} for aid in range(len(comm.agents))]
        first_agent = options[0]["value"] if options else None
        return soc_fig, load_fig, options, first_agent

    @app.callback(
        Output("agent_soc_graph", "figure"),
        [Input("agent_selector", "value"), Input("n_agents", "value")]
    )
    def update_agent_chart(agent_id, n_agents):
        comm = build_community_from_archetypes(csv_path, n_agents=n_agents)
        res = comm.simulate(time_steps=96, dt_h=0.25, optimise=optimise)
        if agent_id not in res["agents"]:
            return go.Figure()
        ares = res["agents"][agent_id]
        t = res["time"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=ares["soc"]*100, mode="lines", name="SoC %"))
        fig.add_trace(go.Scatter(x=t, y=[100 if m else 0 for m in ares["plug_mask"]], fill="tozeroy", mode="none", fillcolor="rgba(0,200,0,0.2)", name="Plugged in"))
        fig.update_layout(title=f"Agent {agent_id} ({comm.agents[agent_id].archetype_name}) SoC Profile", xaxis_title="Hour", yaxis_title="SoC %")
        return fig

    return app

if __name__ == "__main__":
    app = make_app(optimise=False)
    app.run(host="0.0.0.0", port=8050, debug=True)