# CPU Microarchitecture Simulator (arch_designer)

> Web-based sandbox and CLI for exploring **performance ⇄ power ⇄ energy efficiency** trade-offs in CPU microarchitecture, with optional heterogeneous **big.LITTLE** setups.  
> This repository accompanies the paper: *"CPU Microarchitecture Simulator with Manual Parameter Configuration: A Study of Energy Efficiency and Performance"* (SDaS 2025).

<p align="center">
  <img src="docs/figures/page5_img0.png" alt="manual_input UI" width="70%">
</p>

---

## ✨ Features
- **Streamlit UI** for interactive experiments (CPU, memory/cache, I/O, DVFS, big.LITTLE).
- **Manual parameter control**: core counts/types, frequency, voltage, cache hierarchy, activity factor (α), etc.
- **Key metrics**: throughput (proxy/IPC), power, **performance-per-watt**, temperature; thermal map & topology viz.
- **Lightweight models** for rapid “what‑if” studies and education.
- **CLI tools** for batch runs and genetic-search optimization.
- Data samples and a pretrained efficiency model (`models/efficiency_model.pkl`).

---

## 🧭 Project Structure
```
arch_designer/
  app.py                  # Streamlit web app
  main.py                 # CLI entry (click)
  core/                   # Architecture, simulator, optimizer, visualizer
  configs/                # Example configs
  data/                   # Sample datasets
  models/                 # Pretrained efficiency model
  gem5_runs/              # (Optional) external sim traces / outputs
  .streamlit/config.toml  # UI theme
```
> Key modules: `core/architecture.py`, `core/simulator.py`, `core/optimizer.py`, `core/visualizer.py`.

---

## 🚀 Quick Start

### 1) Environment
- **Python 3.10+** (tested on 3.11/3.12).
- Install system deps (Linux/macOS):
  ```bash
  # Ubuntu/Debian
  sudo apt-get update && sudo apt-get install -y graphviz
  # macOS
  brew install graphviz
  ```

### 2) Install dependencies
Create a virtualenv and install Python packages:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel

pip install streamlit altair matplotlib scipy numpy pandas click joblib paramiko graphviz
```

### 3) Run the web app
```bash
streamlit run app.py
```
Open the URL shown in the terminal. Configure cores, caches, DVFS, and (optionally) **big.LITTLE**, then press **Run** to generate metrics and plots.

<p align="center">
  <img src="docs/figures/page5_img1.png" alt="outputs & topology" width="70%">
</p>

### 4) Command-line usage
Examples:
```bash
# Evaluate a single architecture
python main.py evaluate   --arch-type CPU --pipeline 10 --cache-type Inclusive   --l1 64 --l2 1024 --l3 8192   --cores 8 --freq 2.5 --voltage 1.0 --alpha 0.6

# Genetic algorithm search for Pareto candidates
python main.py search   --pop-size 40 --ngen 30 --cxpb 0.7 --mutpb 0.2
```

---

## 📊 Models & Metrics
- **Dynamic power** ~ *C · V² · f*
- **Energy efficiency**: performance-per-watt
- **Thermal behavior**: simplified model for quick feedback (educational focus)
- Pretrained model: `models/efficiency_model.pkl` (loaded automatically by the UI/CLI).

Example trends observed in the paper:
<p align="center">
  <img src="docs/figures/page7_img2.png" alt="Power vs Frequency" width="45%">
  <img src="docs/figures/page7_img3.png" alt="Efficiency vs Frequency" width="45%">
</p>

---

## 📁 Data & External Simulators
- `data/architectures.csv`, `data/dataset.csv` — sample datasets for quick starts.
- `gem5_runs/` — optional directory for traces/results from external tools (e.g., gem5). The app can load summarized metrics; full cycle-accuracy is **out of scope** for this lightweight sandbox.

---

## 🔧 Configuration
Use `configs/my_cpu_config.py` as a starting point. Typical parameters:
```python
ARCH_TYPE = "CPU"
CORES     = 8
FREQ_GHZ  = 2.5
V_CORE    = 1.0
ALPHA     = 0.6   # activity factor
CACHE     = {{"L1": 64, "L2": 1024, "L3": 8192}}
DVFS      = True
HETERO    = {{"enable": True, "big": {{"cores": 4, "freq": 2.5}},
                         "little": {{"cores": 4, "freq": 1.5}}}}
```
> You can point the UI/CLI to a config file or set parameters interactively/through flags.

---

## 🛠 Development
```bash
# lint & style (optional)
pip install ruff black
ruff .
black .
```

Run unit tests (if added later):
```bash
pytest -q
```

---

## 📚 Citation
If you use this simulator in academic work, please cite:
```
Popilnukha I., Chychuzhko M., Kalejnikov G. 
CPU Microarchitecture Simulator with Manual Parameter Configuration: 
A Study of Energy Efficiency and Performance. SDaS 2025.
```
(Replace with final proceedings details once available.)

---

## 🤝 Contributing
PRs and issues are welcome! Please open an issue for bugs/feature requests or submit a PR with a clear description and minimal repro.

---

## 🔒 License
Specify a license for the repository (e.g., MIT, Apache-2.0). If unsure, we recommend MIT for research/education repos.
