# Satellite Network Routing using Q-Learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation of reinforcement learning-based routing algorithms for Satellite-Terrestrial Integrated Networks (STINs). This project focuses on optimizing routing paths in dynamic LEO satellite constellations using **QLRA** and **SQLRA** as proposed by Yin et al. (2021).

---

## 🌟 Overview

LEO satellite networks present unique routing challenges due to high mobility, frequent topology changes, and varying link qualities. This project implements and compares three routing strategies across a 66-node constellation with 3 dynamic time slices:

| Algorithm | Description |
|-----------|-------------|
| **Dijkstra** | Shortest-path baseline (cost = 1 − reward) |
| **QLRA** | Q-Learning routing with ε-greedy exploration (60 episodes) |
| **SQLRA** | Speed-Up QLRA via BFS back-to-front Q-update (30 episodes — 2× faster) |

---

## 📂 Project Structure

```text
.
├── Satellite_QL_Routing.ipynb   # Main analysis notebook (all cells verified)
├── data1.xlsx                   # Time slice 1 — 906 directed ISLs
├── data2.xlsx                   # Time slice 2 — 975 directed ISLs
├── data3.xlsx                   # Time slice 3 — 965 directed ISLs
├── convergence_speed.png        # QLRA vs SQLRA convergence plot
├── performance_same_pair.png    # 4-metric comparison (5–20 users)
├── RL/
│   ├── env.py                   # Satellite network MDP environment
│   ├── rl.py                    # Core DQN agent (reference)
│   ├── config.py                # Hyperparameter configuration
│   └── train.py                 # Training loop
├── rlrouting/
│   ├── ql.py                    # Q-Learning base implementation
│   └── routing.py               # Routing logic
└── Topologies/
    ├── generate_matrices.py     # Adjacency matrix generator
    └── generate_nodes_topo.py   # Node & ISL configuration
```

---

## 📊 Dataset

- **Constellation:** 66 LEO satellites — 11 orbits × 6 satellites per orbit
- **Time Slices:** 3 snapshots (906 / 975 / 965 directed inter-satellite links)
- **Adjacency matrices:** Binary 66×66 matrices in `data1/2/3.xlsx`
- **Synthetic link attributes** (randomly assigned, `np.random.seed(42)`):

| Attribute | Range | Unit |
|-----------|-------|------|
| Bandwidth | 10 – 100 | Mbps |
| Delay | 1 – 20 | ms |
| BER | 1×10⁻⁷ – 9×10⁻⁷ | — |
| Visible Time | 1.0 – 3.0 | min |

---

## 🧮 Reward Function

Based on Yin et al. (2021), Equation 22. Min-max normalized per metric:

$$R = \theta \cdot r(b) + \beta \cdot r(d) + \lambda \cdot r(e) + \omega \cdot r(t)$$

AHP weights (from paper):

| θ (Bandwidth) | β (Delay) | λ (BER) | ω (Visible Time) |
|---|---|---|---|
| 0.30 | 0.15 | 0.18 | 0.37 |

---

## 📈 Results

### Single Pair: Satellite1101 → Satellite1611

| Algorithm | Hops | BW (Mbps) | Delay (ms) | BER (×10⁻⁷) | VT (min) |
|-----------|------|-----------|------------|-------------|---------|
| Dijkstra  | 4    | 50.03     | 14.09      | 2.75        | 2.57    |
| QLRA      | 29   | 64.44     | 321.74     | 4.98        | 1.15    |
| **SQLRA** | 32   | **74.49** | 222.91     | **4.01**    | **1.74** |

**Key findings:**
- SQLRA achieves **49% higher bandwidth** than Dijkstra (74.49 vs 50.03 Mbps)
- SQLRA has **lower BER than QLRA** — better link quality selection
- SQLRA converges in **30 episodes vs QLRA's 60** (2× faster — Paper Figure 8)
- Dijkstra has lowest delay (4 hops only) but ignores BW/BER/Visible Time

### Convergence Speed
![Convergence Speed](convergence_speed.png)

### Multi-User Throughput (5–20 users, same source-destination pair)

| Users | Dijkstra (Mbps) | QLRA (Mbps) | SQLRA (Mbps) |
|-------|----------------|-------------|--------------|
| 5     | 48.43          | 60.78       | 69.11        |
| 10    | 47.23          | 60.63       | 66.21        |
| 15    | 45.63          | 60.83       | 61.19        |
| 20    | 42.58          | 59.51       | 59.76        |

SQLRA consistently outperforms across all user loads.

![Performance Comparison](performance_same_pair.png)

---

## 🔄 Workflow

1. **Load** adjacency matrices → build `networkx` directed graph
2. **Assign** random link weights (BW, Delay, BER, Visible Time)
3. **Compute** per-link reward using AHP-weighted formula
4. **Train** QLRA (ε-greedy, 60 ep) and SQLRA (BFS back-to-front, 30 ep)
5. **Extract** optimal path greedily from converged Q-table
6. **Evaluate** against Dijkstra across 5–20 concurrent users

---

## 🚀 Getting Started

```bash
git clone https://github.com/bytecraft17/Satellite-QL-Routing.git
cd Satellite-QL-Routing
pip install networkx pandas openpyxl numpy matplotlib
jupyter notebook Satellite_QL_Routing.ipynb
```

---

## 📚 Reference

Yin, Y., Huang, C., Wu, D., Huang, S., Abbas Ashraf, M. W., & Guo, Q. (2021).
*Reinforcement Learning-Based Routing Algorithm in Satellite-Terrestrial Integrated Networks.*
Wireless Communications and Mobile Computing, Hindawi/Wiley.
DOI: [10.1155/2021/3759631](https://doi.org/10.1155/2021/3759631)

---

## 📄 License

MIT License
