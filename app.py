import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import paramiko
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import graphviz as gv
import sys
from pathlib import Path
# ensure local plugins are importable
sys.path.append(str(Path(__file__).parent))


from deap import base, creator, tools
from sklearn.ensemble import RandomForestRegressor

from core.architecture import Architecture
from core.simulator import Simulator

# === Helper Functions ===

def calc_thermal(power_w: float, cooling: str):
    """
    Generate a 60×60 thermal map and return (figure, max_temperature).
    """
    Rth = {"air": 0.1, "liquid": 0.05, "passive": 0.5}[cooling]
    dT = power_w * Rth
    ambient = 25.0
    H, W = 60, 60
    grid = np.full((H, W), ambient, dtype=float)
    spots = [
        (int(H * 0.3), int(W * 0.3)),
        (int(H * 0.3), int(W * 0.7)),
        (int(H * 0.7), int(W * 0.3)),
        (int(H * 0.7), int(W * 0.7)),
    ]
    for r, c in spots:
        grid[r, c] += dT
    grid = gaussian_filter(grid, sigma=7)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(grid, cmap="inferno", origin="lower")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)", rotation=270, labelpad=15)
    return fig, grid.max()


def draw_topology(arch: Architecture):
    """Draw Graphviz topology based on architecture parameters."""
    dot = gv.Digraph(format="png")
    dot.attr(rankdir="TB", nodesep="0.3", ranksep="0.2")
    dot.node("CORE", arch.arch_type, shape="oval", style="filled", fillcolor="#FFF")
    dot.node("F", f"Fetch Stages: {arch.pipeline_stages}", shape="box")
    dot.node("E", f"Exec Units: {arch.compute_units}", shape="box")
    dot.node("C", f"Cache {arch.cache_type}\n{arch.cache_size} MB", shape="box")
    phys = (
        f"Transistors: {arch.transistor_count}M\n"
        f"Die Area: {arch.die_area_mm2:.1f} mm²\n"
        f"Vdd: {arch.supply_voltage:.2f} V\n"
        f"Fmax: {arch.max_frequency_ghz:.2f} GHz\n"
        f"Cooling: {arch.cooling_type}"
    )
    dot.node("P", phys, shape="box")
    dot.edges([("CORE","F"), ("CORE","E"), ("F","C"), ("E","P")])
    return dot

# === Page Configuration & White Theme ===
st.set_page_config(page_title="Architecture Designer Sandbox", layout="wide")
st.markdown(
    """
    <style>
    .appview-container, .main, body {background-color: white !important; color: black !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("Architecture Designer Sandbox")

tabs = st.tabs([
    "Manual", "Random", "ML Evaluation", "GA",
    "Dataset Gen", "Training", "Local Sim", "SSH Sim"
])

# === 1. Manual Input ===
with tabs[0]:
    st.header("1. Manual Input")

    # --- Manual Parameters UI ---
    with st.sidebar:
        st.subheader("🧮 CPU Mode & Frequency")
        cpu_mode = st.selectbox("CPU Mode", ["Base Frequency","Turbo Frequency","Power Save"], key="cpu_mode")
        freq = st.number_input("Frequency (GHz)", 0.5, 6.0, 2.0, step=0.1, key="freq")
        dvfs = st.checkbox("Enable DVFS (dynamic V_core)", key="dvfs")

        st.subheader("⚙️ Core Configuration")
        base_cores = st.number_input("Base Cores", 1, 64, 8, key="base_cores")

        if st.checkbox("Enable big.LITTLE (hetero cores)", key="het_en"):
            big_cores = st.number_input("# Big Cores", 1, 32, 4, key="big_cores")
            big_freq = st.number_input("Big Core Freq (GHz)", 0.5, 6.0, 2.5, step=0.1, key="big_freq")
            little_cores = st.number_input("# Little Cores", 1, 32, 4, key="little_cores")
            little_freq = st.number_input("Little Core Freq (GHz)", 0.5, 6.0, 1.5, step=0.1, key="little_freq")
        else:
            big_cores = 0; big_freq=0; little_cores=0; little_freq=0

        st.subheader("🔧 Memory Subsystem")
        working_set = st.number_input("Working Set Size (MB)", 16, 1024, 128, key="ws")
        mem_freq = st.number_input("Memory Frequency (MT/s)", 800, 6400, 3200, key="memf")
        bus_width = st.selectbox("Bus Width (bit)", [32, 64, 128], key="busw")
        mem_bound = st.slider("Memory-bound Fraction", 0.0, 1.0, 0.2, key="mbound")
        mem_profile = st.selectbox("Memory Profile", ["mixed","compute","memory"], key="mprof")

        st.subheader("🖥️ I/O Subsystem")
        io_enable = st.checkbox("Include I/O power" , key="io_en")

        st.subheader("🏁 Workload Properties")
        instr_set = st.multiselect("Instruction Set(s)", ["x86-64","SSE4.2","AVX2","FMA3","AVX512","SVE"], default=["x86-64","AVX2"], key="iset")
        alpha = st.number_input("Activity Factor α", 0.0, 1.0, 0.1, step=0.01, key="alpha")
        ileak = st.number_input("Leakage (A per core)", 1e-8, 1e-3, 1e-6, format="%.1e", key="ileak")

        simulate = st.button("▶ Simulate", key="m_run")

    if simulate:
        # build Architecture using all parameters
        arch = Architecture(
            arch_type="CPU", pipeline_stages=0, cache_type="Inclusive", cache_size=1,
            compute_units=base_cores + big_cores + little_cores,
            branch_predictor="Dynamic", out_of_order=True,
            transistor_count=500, die_area_mm2=150.0,
            supply_voltage=1.0, max_frequency_ghz=freq,
            cooling_type="air"
        )
        res = Simulator().simulate(arch)
        fig, tmax = calc_thermal(res["energy"], "air")
        topo = draw_topology(arch)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("📈 Metrics")
            st.metric("IPC", f"{res['ipc']:.3f}")
            st.metric("Power (W)", f"{res['energy']:.2f}")
            st.metric("Perf/W", f"{res['perf_per_watt']:.3f}")
        with c2:
            st.subheader("🌡️ Thermal Map")
            st.pyplot(fig, use_container_width=True)
            st.caption(f"Max Temp: {tmax:.1f} °C")
        with c3:
            st.subheader("🗺️ Topology")
            st.graphviz_chart(topo, use_container_width=True)

        # --- Interactive Graphs ---
        st.markdown("---")
        st.subheader("📊 Interactive Dependencies")

        sim2 = Simulator()
        # Sweep Frequency
        freqs = np.linspace(0.5, 6.0, 12)
        data_freq = []
        for fval in freqs:
            a2 = Architecture(
                arch_type="CPU", pipeline_stages=0, cache_type="Inclusive", cache_size=1,
                compute_units=base_cores + big_cores + little_cores,
                branch_predictor="Dynamic", out_of_order=True,
                transistor_count=500, die_area_mm2=150.0,
                supply_voltage=1.0, max_frequency_ghz=fval,
                cooling_type="air"
            )
            m2 = sim2.simulate(a2)
            data_freq.append({"Frequency (GHz)": fval, "Power (W)": m2["energy"], "Efficiency": m2["perf_per_watt"]})
        df_freq = pd.DataFrame(data_freq)
        chart_power = (
            alt.Chart(df_freq)
               .mark_line(point=True)
               .encode(x="Frequency (GHz):Q", y="Power (W):Q")
               .properties(width=400, height=300, title="Power vs Frequency")
        )
        chart_eff = (
            alt.Chart(df_freq)
               .mark_line(point=True)
               .encode(x="Frequency (GHz):Q", y="Efficiency:Q")
               .properties(width=400, height=300, title="Efficiency vs Frequency")
        )
        st.altair_chart(chart_power, use_container_width=False)
        st.altair_chart(chart_eff, use_container_width=False)

        # Sweep Exec Units vs Die Temp
        # sweep across all possible exec unit counts
        max_units = base_cores + big_cores + little_cores if (big_cores + little_cores) > 0 else base_cores
        cores_list = list(range(1, max_units+1))
        data_cores = []
        sim_temp = Simulator()
        for cval in cores_list:
            a3 = Architecture(
                arch_type="CPU",
                pipeline_stages=0,
                cache_type="Inclusive",
                cache_size=1,
                compute_units=cval,
                branch_predictor="Dynamic",
                out_of_order=True,
                transistor_count=500,
                die_area_mm2=150.0,
                supply_voltage=1.0,
                max_frequency_ghz=freq,
                cooling_type="air"
            )
            m3 = sim_temp.simulate(a3)
            _, tpeak3 = calc_thermal(m3["energy"], "air")
            data_cores.append({"Exec Units": cval, "T_die (°C)": tpeak3})
        df_cores = pd.DataFrame(data_cores)
        chart_temp = (
            alt.Chart(df_cores)
               .mark_line(point=True, color="red")
               .encode(x="Exec Units:Q", y="T_die (°C):Q")
               .properties(width=400, height=300, title="Die Temp vs Exec Units")
        )
        st.altair_chart(chart_temp, use_container_width=False)

        # Sweep Activity Factor vs Efficiency
        alphas = np.linspace(0.0, 1.0, 11)
        data_alpha = []
        for aval in alphas:
            a4 = Architecture(
                arch_type="CPU",
                pipeline_stages=0,
                cache_type="Inclusive",
                cache_size=1,
                compute_units=base_cores,
                branch_predictor="Dynamic",
                out_of_order=True,
                transistor_count=500,
                die_area_mm2=150.0,
                supply_voltage=1.0,
                max_frequency_ghz=freq,
                cooling_type="air"
            )
            # adjust activity and leakage directly
            setattr(a4, 'alpha', aval)
            setattr(a4, 'ileak', ileak)
            m4 = sim2.simulate(a4)
            data_alpha.append({"Alpha": aval, "Efficiency": m4.get("perf_per_watt", 0)})
        df_alpha = pd.DataFrame(data_alpha)
        chart_alpha = (
            alt.Chart(df_alpha)
               .mark_line(point=True, color="green")
               .encode(x="Alpha:Q", y="Efficiency:Q")
               .properties(width=400, height=300, title="Efficiency vs Activity Factor")
        )
        st.altair_chart(chart_alpha, use_container_width=False)

# === 2. Random Generation === Random Generation === Random Generation === Random Generation === Random Generation === Random Generation ===
with tabs[1]:
    st.header("2. Random Generation")
    count = st.number_input("Number of Architectures", 1, 100, 5)
    if st.button("Generate"):
        sim = Simulator()
        sims = []
        for _ in range(count):
            a = Architecture.random_architecture()
            m = sim.simulate(a)
            sims.append({**a.to_dict(), **m})
        df = pd.DataFrame(sims)
        st.dataframe(df)
        st.download_button("Download JSON", df.to_json(orient="records"), "random.json")

# === 3. ML-Based Estimation ===
with tabs[2]:
    st.header("3. ML-Based Estimation")
    model_path = st.text_input("Model Path", "models/efficiency_model.pkl")
    if st.button("Evaluate ML"):
        sim_ml = Simulator(ml_model_path=model_path)
        arch = Architecture.random_architecture()
        m = sim_ml.simulate(arch)
        st.json(arch.to_dict())
        st.json(m)

# === 4. Genetic Algorithm (GA) ===
with tabs[3]:
    st.header("4. Genetic Algorithm")
    pop_size = st.number_input("Population Size", 10, 500, 50)
    generations = st.number_input("Generations", 1, 200, 40)
    crossover_prob = st.slider("Crossover Probability", 0.0, 1.0, 0.5)
    mutation_prob = st.slider("Mutation Probability", 0.0, 1.0, 0.2)
    if st.button("Run GA"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        def create_individual():
            return [random.randrange(5),
                    random.randint(4, 20),
                    random.choice([1, 2, 4, 8, 16]),
                    random.randrange(3),
                    random.randint(1, 64),
                    random.randrange(2),
                    random.randrange(2)]
        toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
