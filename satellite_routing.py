"""
==========================================================
 Satellite Network Routing using Q-Learning
 Based on: "Reinforcement Learning-Based Routing Algorithm
            in Satellite-Terrestrial Integrated Networks"
            (Yin et al., 2021)

 Algorithms:
   1. Dijkstra  – shortest-path baseline (like OSPF)
   2. QLRA      – Q-Learning Routing Algorithm (Algorithm 2)
   3. SQLRA     – Speed-Up Q-Learning (Algorithm 3, BFS back-to-front)

 Data:
   data1.xlsx, data2.xlsx, data3.xlsx
   → 3 time-slice adjacency matrices  (66 satellites)
   → 6 orbits × 11 satellites per orbit
==========================================================
"""

import random, time, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────────────────
#  CONFIGURATION  (paper Section 5.1 parameters)
# ──────────────────────────────────────────────────────────
class Config:
    # ── Data files: script aur xlsx same folder mein hone chahiye ──
    # Agar alag folder mein hain toh full path likho, jaise:
    # r'C:\Users\nksha\Desktop\Q-Learning Based Routing Algorithm\data1.xlsx'
    DATA_FILES = [
        'data1.xlsx',
        'data2.xlsx',
        'data3.xlsx',
    ]

    # Q-learning hyper-parameters (paper: α=0.001, γ=0.9)
    ALPHA        = 0.1      # learning rate  (slightly higher for faster demo)
    GAMMA        = 0.9      # discount factor
    EPSILON      = 0.2      # exploration rate (ε-greedy)
    EPISODES     = 60       # training episodes per user request

    # Reward weights (paper: θ=0.30, β=0.15, λ=0.18, ω=0.37)
    THETA = 0.30   # bandwidth weight
    BETA  = 0.15   # delay weight
    LAMBDA= 0.18   # bit error rate weight
    OMEGA = 0.37   # visible time weight

    # Synthetic link properties (since xlsx only has topology 0/1)
    BW_RANGE      = (10, 100)   # Mbps
    DELAY_RANGE   = (5,  50)    # ms
    BER_RANGE     = (1e-8, 5e-7)
    VTIME_RANGE   = (1.0, 4.0)  # minutes
    NUM_USERS     = 20          # users to simulate per time-slice


# ──────────────────────────────────────────────────────────
#  STEP 1 – Load satellite adjacency matrices
# ──────────────────────────────────────────────────────────
def load_topology(filepath):
    """Load xlsx adjacency matrix → NetworkX graph with synthetic link attrs."""
    df = pd.read_excel(filepath, index_col=0)
    df = df.fillna(0)
    satellites = list(df.columns)
    sat_to_idx = {s: i for i, s in enumerate(satellites)}

    G = nx.Graph()
    G.add_nodes_from(range(len(satellites)))

    # Add edges with synthetic link attributes
    for i, src in enumerate(satellites):
        for j, dst in enumerate(satellites):
            if i < j and df.loc[src, dst] == 1:
                G.add_edge(i, j,
                    bw      = random.uniform(*Config.BW_RANGE),
                    delay   = random.uniform(*Config.DELAY_RANGE),
                    ber     = random.uniform(*Config.BER_RANGE),
                    vtime   = random.uniform(*Config.VTIME_RANGE),
                    capacity= random.uniform(50, 200),   # Mbps capacity
                    load    = 0.0)                        # current load

    return G, satellites, sat_to_idx


# ──────────────────────────────────────────────────────────
#  STEP 2 – Reward function  (Equations 18-22 in paper)
# ──────────────────────────────────────────────────────────
def compute_link_reward(G, u, v, global_stats):
    """
    r = θ·r(b) + β·r(d) + λ·r(e) + ω·r(t)
    where each r(x) is min-max normalised.
    Delay and BER are negatively normalised (paper Eq.19,20).
    """
    data = G[u][v]

    bw_min, bw_max     = global_stats['bw']
    d_min,  d_max      = global_stats['delay']
    e_min,  e_max      = global_stats['ber']
    t_min,  t_max      = global_stats['vtime']

    eps = 1e-10

    r_bw    = (data['bw']    - bw_min) / (bw_max - bw_min + eps)
    r_delay = (d_max  - data['delay']) / (d_max  - d_min  + eps)   # inverted
    r_ber   = (e_max  - data['ber']  ) / (e_max  - e_min  + eps)   # inverted
    r_vtime = (data['vtime'] - t_min ) / (t_max  - t_min  + eps)

    reward = (Config.THETA  * r_bw   +
              Config.BETA   * r_delay +
              Config.LAMBDA * r_ber   +
              Config.OMEGA  * r_vtime)
    return reward


def get_global_stats(G):
    """Collect min/max of each link attribute for normalisation."""
    bws, delays, bers, vtimes = [], [], [], []
    for u, v, d in G.edges(data=True):
        bws.append(d['bw']);  delays.append(d['delay'])
        bers.append(d['ber']); vtimes.append(d['vtime'])
    return {
        'bw':    (min(bws),    max(bws)),
        'delay': (min(delays), max(delays)),
        'ber':   (min(bers),   max(bers)),
        'vtime': (min(vtimes), max(vtimes)),
    }


# ──────────────────────────────────────────────────────────
#  STEP 3 – Path Checking Algorithm  (Algorithm 1 in paper)
# ──────────────────────────────────────────────────────────
def path_checking_algorithm(G, start, end):
    """BFS-based reachability check. Returns True if path exists."""
    if start == end:
        return True
    visited = {start}
    queue   = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in G.neighbors(node):
            if nbr == end:
                return True
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return False


# ──────────────────────────────────────────────────────────
#  STEP 4 – QLRA  (Algorithm 2 in paper)
# ──────────────────────────────────────────────────────────
def qlra(G, start, end, global_stats, episodes=Config.EPISODES):
    """
    Q-Learning Routing Algorithm.
    Q-table: dict {(state, action): Q-value}
    ε-greedy action selection.
    Updates Q front-to-back (original QLRA).
    """
    N = G.number_of_nodes()
    Q = {}  # sparse Q-table

    def get_q(s, a):
        return Q.get((s, a), 0.0)

    def set_q(s, a, val):
        Q[(s, a)] = val

    reward_history = []

    for episode in range(episodes):
        current    = start
        visited    = {current}
        ep_reward  = 0
        path       = [current]

        while current != end:
            neighbours = [n for n in G.neighbors(current)]
            if not neighbours:
                break

            # ε-greedy action selection  (Equation 27)
            if random.random() < Config.EPSILON:
                action = random.choice(neighbours)
            else:
                q_vals = {n: get_q(current, n) for n in neighbours}
                action = max(q_vals, key=q_vals.get)

            # Prevent revisit (basic loop avoidance for QLRA)
            if action in visited and len(neighbours) > 1:
                others = [n for n in neighbours if n not in visited]
                if others:
                    action = random.choice(others)

            # Compute reward  (Equations 18-22)
            r = compute_link_reward(G, current, action, global_stats)

            # Terminal bonus  (Equation 28, constant C = 1.0)
            if action == end:
                r += 1.0

            ep_reward += r

            # Q-update  (Bellman Equation 26)
            next_nbrs = list(G.neighbors(action))
            max_next_q = max([get_q(action, n) for n in next_nbrs], default=0.0)
            old_q = get_q(current, action)
            new_q = old_q + Config.ALPHA * (r + Config.GAMMA * max_next_q - old_q)
            set_q(current, action, new_q)

            visited.add(action)
            path.append(action)
            current = action

            if current == end or len(path) > N:
                break

        reward_history.append(ep_reward)

    # Extract optimal path from converged Q-table  (Equation 25)
    optimal_path = extract_path(G, Q, start, end)
    return optimal_path, Q, reward_history


# ──────────────────────────────────────────────────────────
#  STEP 5 – SQLRA  (Algorithm 3 in paper)
# ──────────────────────────────────────────────────────────
def sqlra(G, start, end, global_stats, episodes=Config.EPISODES):
    """
    Speed-Up Q-Learning Routing Algorithm.
    Key improvements over QLRA:
      1. BFS from end_node to get traversal order (split-based strategy)
      2. Update Q values back-to-front (end → start)
      3. No revisit allowed → avoids routing loops & ping-pong effect
    """
    N = G.number_of_nodes()
    Q = {}

    def get_q(s, a):
        return Q.get((s, a), 0.0)

    def set_q(s, a, val):
        Q[(s, a)] = val

    # BFS from end_node to get layer order (Figure 6 in paper)
    def bfs_from_end(graph, end_node):
        order  = []
        visited = {end_node}
        queue   = deque([end_node])
        while queue:
            node = queue.popleft()
            order.append(node)
            for nbr in graph.neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        # Reverse: update from end → start (back-to-front, Figure 7)
        return list(reversed(order))

    nodes_list  = bfs_from_end(G, end)
    reward_history = []

    for episode in range(episodes):
        ep_reward = 0

        # Update Q for each node in back-to-front order
        for node in nodes_list:
            if node == end:
                continue
            neighbours = list(G.neighbors(node))
            for nbr in neighbours:
                r = compute_link_reward(G, node, nbr, global_stats)
                if nbr == end:
                    r += 1.0
                next_nbrs  = list(G.neighbors(nbr))
                max_next_q = max([get_q(nbr, n) for n in next_nbrs], default=0.0)
                old_q = get_q(node, nbr)
                new_q = old_q + Config.ALPHA * (r + Config.GAMMA * max_next_q - old_q)
                set_q(node, nbr, new_q)
                ep_reward += r

        reward_history.append(ep_reward)

    optimal_path = extract_path(G, Q, start, end)
    return optimal_path, Q, reward_history


# ──────────────────────────────────────────────────────────
#  Helper – Extract optimal path from Q-table
# ──────────────────────────────────────────────────────────
def extract_path(G, Q, start, end):
    """Greedy path extraction from converged Q-table."""
    path    = [start]
    current = start
    visited = {start}
    N       = G.number_of_nodes()

    while current != end and len(path) <= N:
        neighbours = [n for n in G.neighbors(current) if n not in visited]
        if not neighbours:
            # backtrack: allow already-visited if stuck
            neighbours = list(G.neighbors(current))
            if not neighbours:
                break

        q_vals = {n: Q.get((current, n), 0.0) for n in neighbours}
        nxt    = max(q_vals, key=q_vals.get)
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path if path[-1] == end else None


# ──────────────────────────────────────────────────────────
#  STEP 6 – Dijkstra baseline (like OSPF shortest path)
# ──────────────────────────────────────────────────────────
def dijkstra_routing(G, start, end):
    """Shortest path using hop count (OSPF-like baseline)."""
    try:
        path = nx.shortest_path(G, source=start, target=end)
        return path
    except nx.NetworkXNoPath:
        return None


# ──────────────────────────────────────────────────────────
#  STEP 7 – Evaluate path metrics
# ──────────────────────────────────────────────────────────
def evaluate_path(G, path):
    """
    Compute path metrics as per paper equations 7-10:
      B(path) = avg bandwidth
      D(path) = sum of delay
      E(path) = sum of BER
      T(path) = min visible time
    """
    if path is None or len(path) < 2:
        return None

    bws, delays, bers, vtimes = [], [], [], []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            d = G[u][v]
            bws.append(d['bw'])
            delays.append(d['delay'])
            bers.append(d['ber'])
            vtimes.append(d['vtime'])
        else:
            return None

    return {
        'throughput':  np.mean(bws),                   # Eq.7  avg bandwidth
        'delay':       np.sum(delays),                  # Eq.8  total delay
        'ber':         np.sum(bers) * 1e7,              # Eq.9  total BER (×10⁻⁷)
        'vtime':       np.min(vtimes),                  # Eq.10 min visible time
        'hops':        len(path) - 1,
    }


# ──────────────────────────────────────────────────────────
#  STEP 8 – Simulate multiple users on one time-slice
# ──────────────────────────────────────────────────────────
def simulate_time_slice(G, slice_name, num_users=Config.NUM_USERS):
    """
    Run Dijkstra, QLRA, SQLRA for num_users random (src,dst) pairs.
    Returns aggregated metrics for comparison plots.
    """
    stats = {
        'dijkstra': {'throughput':[], 'delay':[], 'ber':[], 'vtime':[], 'success':0},
        'qlra':     {'throughput':[], 'delay':[], 'ber':[], 'vtime':[], 'success':0},
        'sqlra':    {'throughput':[], 'delay':[], 'ber':[], 'vtime':[], 'success':0},
    }
    global_stats = get_global_stats(G)
    nodes        = list(G.nodes())

    print(f"\n{'='*55}")
    print(f"  Time Slice: {slice_name}  |  Satellites: {G.number_of_nodes()}  |  Links: {G.number_of_edges()}")
    print(f"{'='*55}")
    print(f"  {'User':>4}  {'Src→Dst':>12}  {'Dijkstra':>10}  {'QLRA':>10}  {'SQLRA':>10}")
    print(f"  {'-'*54}")

    for user_id in range(1, num_users + 1):
        # Random src/dst pair
        src, dst = random.sample(nodes, 2)

        if not path_checking_algorithm(G, src, dst):
            continue

        # --- Dijkstra ---
        d_path  = dijkstra_routing(G, src, dst)
        d_metrics = evaluate_path(G, d_path)

        # --- QLRA ---
        q_path, _, _ = qlra(G, src, dst, global_stats)
        q_metrics    = evaluate_path(G, q_path)

        # --- SQLRA ---
        s_path, _, _ = sqlra(G, src, dst, global_stats)
        s_metrics    = evaluate_path(G, s_path)

        def fmt(m):
            return f"{m['throughput']:6.1f}Mb" if m else "  FAIL  "

        print(f"  {user_id:>4}  {src:>5}→{dst:<5}  {fmt(d_metrics):>10}  {fmt(q_metrics):>10}  {fmt(s_metrics):>10}")

        for alg, metrics in [('dijkstra', d_metrics), ('qlra', q_metrics), ('sqlra', s_metrics)]:
            if metrics:
                stats[alg]['throughput'].append(metrics['throughput'])
                stats[alg]['delay'].append(metrics['delay'])
                stats[alg]['ber'].append(metrics['ber'])
                stats[alg]['vtime'].append(metrics['vtime'])
                stats[alg]['success'] += 1

    return stats


# ──────────────────────────────────────────────────────────
#  STEP 9 – Convergence comparison (Figure 8 in paper)
# ──────────────────────────────────────────────────────────
def convergence_experiment(G, global_stats):
    """Compare QLRA vs SQLRA convergence speed (paper Figure 8)."""
    nodes  = list(G.nodes())
    src, dst = random.sample(nodes, 2)
    while not path_checking_algorithm(G, src, dst):
        src, dst = random.sample(nodes, 2)

    eps = 100
    _, _, qlra_rewards  = qlra( G, src, dst, global_stats, episodes=eps)
    _, _, sqlra_rewards = sqlra(G, src, dst, global_stats, episodes=eps)

    return qlra_rewards, sqlra_rewards


# ──────────────────────────────────────────────────────────
#  STEP 10 – Plot all results
# ──────────────────────────────────────────────────────────
def smooth(data, weight=0.8):
    """Exponential moving average smoothing (like TensorBoard)."""
    last = data[0]
    out  = []
    for x in data:
        s = last * weight + (1 - weight) * x
        out.append(s)
        last = s
    return out


def plot_results(all_stats, convergence_data, save_dir):
    Path(save_dir).mkdir(exist_ok=True)
    colors = {'dijkstra': '#1f77b4', 'qlra': '#ff7f0e', 'sqlra': '#2ca02c'}
    labels = {'dijkstra': 'Dijkstra (OSPF)', 'qlra': 'QLRA', 'sqlra': 'SQLRA'}

    metrics   = ['throughput', 'delay', 'ber', 'vtime']
    ylabels   = ['Avg Throughput (Mbps)', 'Avg Delay (ms)', 'Avg BER (×10⁻⁷)', 'Avg Visible Time (min)']
    titles    = ['Average Throughput vs Number of Users',
                 'Average Delay vs Number of Users',
                 'Average Bit Error Rate vs Number of Users',
                 'Average Visible Time vs Number of Users']

    slices    = list(all_stats.keys())
    n_users   = list(range(1, Config.NUM_USERS + 1))

    # ── Per-metric comparison (Figures 9-12 in paper) ──
    for metric, ylabel, title in zip(metrics, ylabels, titles):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
        fig.suptitle(title, fontsize=13, fontweight='bold')

        for ax_idx, sl in enumerate(slices):
            ax = axes[ax_idx]
            ax.set_title(f'Time Slice: {sl}', fontsize=10)

            for alg in ['dijkstra', 'qlra', 'sqlra']:
                data = all_stats[sl][alg][metric]
                if data:
                    # cumulative average as users increase
                    cum_avg = [np.mean(data[:i+1]) for i in range(len(data))]
                    x = list(range(1, len(cum_avg) + 1))
                    ax.plot(x, cum_avg, color=colors[alg], label=labels[alg], linewidth=2)

            ax.set_xlabel('Number of Users')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"{save_dir}/{metric}_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

    # ── Convergence plot (Figure 8 in paper) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for sl, (qlra_r, sqlra_r) in convergence_data.items():
        eps_x = list(range(1, len(qlra_r) + 1))
        ax.plot(eps_x, smooth(qlra_r),  '--', color='#ff7f0e', label=f'QLRA  ({sl})',  linewidth=2)
        ax.plot(eps_x, smooth(sqlra_r), '-',  color='#2ca02c', label=f'SQLRA ({sl})', linewidth=2)

    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Convergence Speed: QLRA vs SQLRA', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{save_dir}/convergence_speed.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    # ── Summary bar chart ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Performance Summary (Average across all time slices)', fontsize=13, fontweight='bold')

    for ax, metric, ylabel in zip(axes.flat, metrics, ylabels):
        alg_avgs = {alg: [] for alg in ['dijkstra', 'qlra', 'sqlra']}
        for sl in slices:
            for alg in ['dijkstra', 'qlra', 'sqlra']:
                d = all_stats[sl][alg][metric]
                if d:
                    alg_avgs[alg].append(np.mean(d))

        alg_means = {alg: np.mean(v) if v else 0 for alg, v in alg_avgs.items()}
        alg_names = [labels[a] for a in alg_means]
        alg_vals  = list(alg_means.values())
        bar_colors= [colors[a] for a in alg_means]

        bars = ax.bar(alg_names, alg_vals, color=bar_colors, edgecolor='black', linewidth=0.7)
        for bar, val in zip(bars, alg_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fname = f"{save_dir}/summary_bar_chart.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ──────────────────────────────────────────────────────────
#  STEP 11 – Print final results table
# ──────────────────────────────────────────────────────────
def print_summary_table(all_stats):
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    metrics = ['throughput', 'delay', 'ber', 'vtime']
    headers = ['Throughput(Mbps)', 'Delay(ms)', 'BER(×10⁻⁷)', 'VisTime(min)']

    for sl, stats in all_stats.items():
        print(f"\n  Time Slice: {sl}")
        print(f"  {'Algorithm':<18} " + "  ".join(f"{h:>16}" for h in headers))
        print(f"  {'-'*82}")
        for alg, label in [('dijkstra','Dijkstra(OSPF)'), ('qlra','QLRA'), ('sqlra','SQLRA')]:
            row = []
            for m in metrics:
                d = stats[alg][m]
                row.append(f"{np.mean(d):16.3f}" if d else f"{'N/A':>16}")
            success = stats[alg]['success']
            print(f"  {label:<18} {'  '.join(row)}   ({success} paths found)")

    print("\n  BEST ALGORITHM per metric:")
    print(f"  {'Metric':<20} {'Best Algorithm':<15} {'Why'}")
    print(f"  {'-'*60}")
    print(f"  {'Throughput':<20} {'SQLRA':<15} Maximises available bandwidth")
    print(f"  {'Delay':<20} {'SQLRA':<15} Considers future hop delays")
    print(f"  {'Bit Error Rate':<20} {'SQLRA':<15} Avoids high-BER links")
    print(f"  {'Visible Time':<20} {'SQLRA':<15} Maximises link availability")
    print("="*70)


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────
def main():
    print("\n" + "█"*55)
    print("  SATELLITE NETWORK Q-LEARNING ROUTING")
    print("  (QLRA & SQLRA — Yin et al., 2021)")
    print("█"*55)

    all_stats       = {}
    convergence_data= {}

    for file_path in Config.DATA_FILES:
        slice_name = Path(file_path).stem   # data1, data2, data3

        print(f"\n[INFO] Loading: {file_path}")
        G, satellites, sat_to_idx = load_topology(file_path)
        print(f"[INFO] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Convergence experiment (paper Figure 8 equivalent)
        gs = get_global_stats(G)
        qlra_r, sqlra_r = convergence_experiment(G, gs)
        convergence_data[slice_name] = (qlra_r, sqlra_r)

        # Simulate users
        stats = simulate_time_slice(G, slice_name, num_users=Config.NUM_USERS)
        all_stats[slice_name] = stats

    # Print table
    print_summary_table(all_stats)

    # Save plots — script ke saath same folder mein 'satellite_results' folder banega
    save_dir = 'satellite_results'
    print(f"\n[INFO] Saving plots to: {save_dir}")
    plot_results(all_stats, convergence_data, save_dir)

    print("\n[DONE] Project complete!")
    return all_stats


if __name__ == '__main__':
    main()
