# ADRC Pro Tuner

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

 

**A Universal Automatic Tuning and Analysis Platform for Linear and Nonlinear Dynamical Systems**

 

*Active Disturbance Rejection Control — from plant specification to validated closed loop design in one click*
The platform was developed by **HANI NARIMENE** ( a control systems engineer ) .

## Contact

**HANI Narimene**  
📧 hnnarimene@gmail.com
---

[Features](#features) · [Installation](#installation) · [Quick Start](#quick-start) · [Modules](#modules) · [Test Cases](#test-cases) · [References](#references) · [Contact](#contact)

</div>

---

## Overview

**ADRC Pro Tuner** is an open source interactive engineering platform for the automatic design, simulation, and analysis of **Active Disturbance Rejection Control (ADRC)** applied to general SISO dynamical systems. It implements the bandwidth parameterised Linear ADRC (LADRC) framework of Gao (2003) and the Nonlinear ADRC (NLADRC) extension of Han (1998) within a unified Python/Streamlit environment.

 

---

## Features

### Core capabilities

| Feature | Description |
|---|---|
| **3 plant model types** | Transfer function, state-space matrices (A,B,C,D), user-defined nonlinear ODE |
| **LADRC** | Bandwidth-parameterised Linear ADRC for any order n ≥ 1 |
| **NLADRC** | Nonlinear ADRC with Han's `fal()` function, configurable α and δ |
| **Auto-tuner** | Dominant-pole bandwidth scaling — 4 objective presets, zero manual initialisation |
| **RK4 simulation** | Fixed-step fourth-order Runge-Kutta, Δt = 10⁻³ s, configurable disturbances and noise |
| **8 performance metrics** | Rise time, settling time, overshoot, steady-state error, ISE, IAE, ITAE, control energy |
| **PID comparison** | Auto-tuned Ziegler-Nichols PID under identical conditions |
| **ESO visualisation** | Real-time display of observer state estimates and disturbance tracking |
| **Stability analysis** | Closed-loop pole map, eigenvalue computation |
| **MATLAB export** | `.m` script + JSON configuration for Simulink deployment |

### What sets it apart from existing tools

| Feature | ADRC Pro Tuner | pyadrc | ADRC Toolbox (Łakomy et al.) |
|---|---|---|---|
| Interactive GUI | ✅ Web (Streamlit) | ❌ Library only | ✅ Simulink blocks |
| Automatic tuning | ✅ Dominant-pole | ❌ Manual only | ❌ Manual only |
| NLADRC (fal) | ✅ Full implementation | ❌ | ❌ |
| Arbitrary order n | ✅ Any n ≥ 1 | ❌ n ≤ 2 | ✅ |
| Nonlinear ODE plants | ✅ User-defined | ❌ | ❌ |
| PID benchmarking | ✅ 8 metrics | ❌ | ❌ |
| No proprietary software | ✅ Pure Python | ✅ | ❌ Requires MATLAB |

 ## Installation & How to Run

### Step 1 — Install Python 3.12

Download and install Python 3.12 from:  
👉 https://www.python.org/downloads/

> **Windows users:** during installation, check **"Add Python to PATH"**

Verify:
```bash
python --version
# Expected: Python 3.12.x
```

---

### Step 2 — Download the project

**Option A — Clone with Git:**
```bash
git clone https://github.com/hani-narimene/adrc-pro-tuner.git
cd adrc-pro-tuner
```

**Option B — Download ZIP (no Git needed):**
- Click the green **"Code"** button on this page
- Click **"Download ZIP"**
- Extract the folder
- Open a terminal inside the extracted folder

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `numpy`, `scipy`, `matplotlib`  
Takes 1–2 minutes on first install.

---

### Step 4 — Launch the application

```bash
streamlit run app.py
```

The browser opens automatically at:
```
http://localhost:8501
```

> If the browser does not open, copy and paste the URL manually.

---

### Full sequence (copy-paste ready)

```bash
git clone https://github.com/hani-narimene/adrc-pro-tuner.git
cd adrc-pro-tuner
pip install -r requirements.txt
streamlit run app.py
```

---
## Quick Start

### 1 — Define your plant

Select the model type in the left sidebar:

```
Transfer Function:   Numerator: [1]    Denominator: [1, 2, 1]
                     → G(s) = 1/(s+1)²

State-Space:         Enter A, B, C, D matrices directly

Nonlinear ODE:       # Van der Pol (mu=1)
                     dx0 = x[1]
                     dx1 = (1 - x[0]**2)*x[1] - x[0] + u
                     return [dx0, dx1]
```

### 2 — Select controller and tuning objective

```
Controller type:    Linear ADRC (LADRC)  |  Nonlinear ADRC (NLADRC)

Tuning objective:   Fast response        →  ωc = 8·σmax,  k = 8
                    Balanced             →  ωc = 4·σmax,  k = 5
                    Robust               →  ωc = 2.5·σmax, k = 6
                    Noise attenuation    →  ωc = 1.5·σmax, k = 3

Auto-tune:          ✅ recommended — derives (ωc, ωo) automatically
```

### 3 — Configure scenario

```
Reference:      Step | Ramp | Sine | Square
Disturbance:    Step | Sine | Pulse   (amplitude, onset time)
Noise:          Optional Gaussian measurement noise (σ)
Horizon:        T [s]    Step: Δt = 10⁻³ s
```

### 4 — Read results

Navigate the 7 tabs in the main panel:

| Tab | Content |
|---|---|
| 🎯 Response | y(t), r(t), u(t) time histories |
| 🔭 ESO & Disturbance | ẑ(t) tracking, f̂(t) estimate vs true d(t) |
| ⚖️ ADRC vs PID | Overlaid curves + metric comparison table |
| 🧮 Parameters | ωc, ωo, b₀, K, L vectors |
| 🛡️ Stability | Pole-zero map, stability certificate |
| 📤 Export | MATLAB script, JSON, plain-text report |
| 📚 Theory | In-app mathematical reference |

---

## Modules

The platform is structured around five independent modules:

```
Input ──► [Parser] ──► [Tuner] ──► [Engine] ──► [Simulator] ──► [Analysis] ──► Output
                         ▲                                              │
                         └──────────────── iterate ────────────────────┘
```

### Module 1 — Plant Parser
Accepts TF coefficients, SS matrices (A,B,C,D), or a Python ODE expression.
Extracts the dominant pole σ_max and estimates b₀ from the Markov parameters.

### Module 2 — ADRC Design Engine
Implements LADRC (Gao 2003) and NLADRC (Han 1998) for arbitrary order n.

```
ESO gains:        L_i = C(n+1, i) · ωo^i          i = 1,...,n+1
Controller gains: k_i = C(n,   i-1) · ωc^(n-i+1)  i = 1,...,n
```

### Module 3 — Auto-Tuner
```
ωc = β · σ_max        (β from objective preset)
ωo = k · ωc           (k ∈ [3, 10])
```
Iterates automatically if overshoot > 5% or Ts > target.

### Module 4 — Simulation Engine
Fixed-step RK4 co-integrating plant state x(t) and ESO state ẑ(t) simultaneously.
Random seed = 42. Zero initial conditions.

### Module 5 — Analysis and Export
Computes 8 metrics. Generates all plots. Exports:
- `adrc_params.m` — MATLAB/Simulink-ready script
- `adrc_params.json` — machine-readable configuration
- `adrc_report.txt` — structured plain-text report

---

## Test Cases

Validated on 9 test cases — ADRC vs auto-tuned PID:

| # | System | Ts ADRC | Ts PID | δ% ADRC | δ% PID |
|---|---|---|---|---|---|
| 1 | 2nd-order linear (LADRC) | **1.31 s** | 3.10 s | **0.83%** | 4.20% |
| 2 | 1st-order nonlinear (NLADRC) | **1.68 s** | 3.45 s | **2.30%** | 8.50% |
| 3 | 4th-order linear (LADRC) | **2.28 s** | 4.90 s | **5.01%** | 18.7% |
| 4 | 2nd-order, ramp (LADRC) | **2.10 s** | 4.20 s | — | — |
| 5 | 3rd-order, sinusoidal (LADRC) | **1.95 s** | 3.80 s | **1.20%** | 6.30% |
| 6 | 2nd-order, noisy (LADRC) | **1.82 s** | 3.50 s | **1.60%** | 5.80% |
| 7 | 1st-order, fast (LADRC) | **0.58 s** | 1.20 s | **3.40%** | 9.10% |
| 8 | 2nd-order (NLADRC) | **1.44 s** | 3.10 s | **0.61%** | 4.20% |
| 9 | 3rd-order nonlinear (NLADRC) | **2.35 s** | 5.10 s | **4.10%** | 14.3% |
| | **Mean improvement** | **−53%** | | **−74%** | |

---

## ADRC Background

ADRC was introduced by **Han (1998)** and reformulated in bandwidth-parameterised form by **Gao (2003)**. The key idea: treat all unmodelled dynamics, parameter variations, and external disturbances as a single *total disturbance* f(t), estimate it in real time with the Extended State Observer, and cancel it in the control law.

```
Plant:   y^(n) = f(y, ẏ, ..., w, t) + b₀·u

ESO:     ż = A·z + B·u + L·(y − z₁)       ← estimates f(t) as z_{n+1}

Control: u = (u₀ − ẑ_{n+1}) / b₀          ← cancels estimated disturbance
         u₀ = Σ kᵢ·(rᵢ − ẑᵢ)              ← places poles at −ωc
```

Under accurate estimation, the closed loop reduces to a chain of n integrators — independent of plant model and disturbances.

---

## Reproducibility

```
Python version:      3.12
Random seed:         42  (fixed for all noise generation)
Integration step:    Δt = 10⁻³ s  (fixed-step RK4)
Initial conditions:  x(0) = 0,  ẑ(0) = 0  (all zero)
```

---

## References

[1] J. Han, "From PID to active disturbance rejection control," *IEEE Trans. Ind. Electron.*, vol. 56, no. 3, pp. 900–906, 2009.

[2] Z. Gao, "Scaling and bandwidth-parameterization based controller tuning," in *Proc. American Control Conf.*, 2003, pp. 4989–4996.

[3] G. Herbst, "A simulative study on ADRC as a control tool for practitioners," *Electronics*, vol. 2, no. 3, pp. 246–279, 2013.

[4] K. Łakomy et al., "ADRC Toolbox for MATLAB/Simulink," *arXiv:2112.01614*, 2022.

[5] O. Türkcüoğlu, "pyadrc," GitHub, 2021. https://github.com/onguntoglu/pyadrc

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.


---

<div align="center">
<sub>ADRC Pro Tuner · 2026</sub>
</div>
