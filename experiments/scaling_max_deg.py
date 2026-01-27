import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import psutil
import numpy as np
import time
import faircat
import gc

# user-defined parameters
k = 5
d = 2
corr_targets = {}

# M: kxk (moderate homophily)
M = np.array([
    [0.4,  0.15, 0.15, 0.15, 0.15],
    [0.15, 0.4,  0.15, 0.15, 0.15],
    [0.15, 0.15, 0.4,  0.15, 0.15],
    [0.15, 0.15, 0.15, 0.4,  0.15],
    [0.15, 0.15, 0.15, 0.15, 0.4 ],
])

# D: kxk
D = np.array([
    [0.20, 0.05, 0.05, 0.05, 0.05],
    [0.05, 0.20, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.25, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.25, 0.05],
    [0.05, 0.05, 0.05, 0.05, 0.30],
])

# H: (d-1) x k - moderate separation
H = np.array([
    [0.4, 0.1, 0.1, 0.4, 0.1]
])

# Pcg
g = 2     # binary sensitive attribute
k = 5
base = np.array([0.4, 0.15, 0.15, 0.15, 0.15])
Pcg = np.vstack([base, base])   # shape (g, k)
Pcg /= Pcg.sum(axis=1, keepdims=True)

# degree distribution types
dist_type_0 = "powerlaw"
dist_type_1 = "powerlaw"

results = []

z_0 = 20
z_1 = 20

max_deg_multiplier = [2**i for i in range(1, 11)]

process = psutil.Process(os.getpid())

for mult in max_deg_multiplier:
    print(f"\n=== Running z0={z_0}, z1={z_1} | multiplier={mult} ===")
    deg_0 = 2**z_0
    deg_1 = 2**z_1

    n_0 = int(deg_0 / 32)
    n_1 = int(deg_1 / 32)

    avg_deg_0 = deg_0 / n_0
    avg_deg_1 = deg_1 / n_1
    max_deg_0 = int(avg_deg_0 * mult)
    max_deg_1 = int(avg_deg_1 * mult)

    mem_before = process.memory_info().rss / (1024**3)  # GB
    start = time.time()

    # faircat run
    S, X, Label, theta, sorted_attr_group = faircat.faircat(
        n_0, n_1, deg_0, deg_1,
        k, d, max_deg_0, max_deg_1,
        dist_type_0, dist_type_1,
        Pcg, M, D, H,
        att_type="normal",
        corr_targets=corr_targets,
        MAPE=True
    )

    # compute mape
    S = S.tocsr()
    S.setdiag(0)
    if hasattr(S, "eliminate_zeros"):
        S.eliminate_zeros()
    if S.nnz:
        S.data[:] = 1

    theta_actual = np.array(S.sum(axis=1)).ravel().astype(float)
    theta_target = np.asarray(theta, float)
    theta_target = np.clip(theta_target, 1e-12, None)

    mape_total = np.mean(np.abs(theta_actual - theta_target) / theta_target)

    groups = np.asarray(sorted_attr_group)
    mape_0 = np.mean(np.abs(theta_actual[groups == 0] - theta_target[groups == 0]) / theta_target[groups == 0])
    mape_1 = np.mean(np.abs(theta_actual[groups == 1] - theta_target[groups == 1]) / theta_target[groups == 1])

    end = time.time()
    runtime = end - start

    mem_after = process.memory_info().rss / (1024**3)  # GB
    mem_used = max(mem_after - mem_before, 0.0)

    del S, X, Label, theta, sorted_attr_group
    del theta_actual, theta_target, groups
    gc.collect()

    results.append((z_0, z_1, mult, mape_total, mape_0, mape_1, runtime, mem_used))
    print(
        f"MAPE overall: {mape_total:.2e} | g0: {mape_0:.2e} | g1: {mape_1:.2e} | "
        f"Time: {runtime:.1f}s | Mem: {mem_used:.2f} GB"
    )

print("\n===== MAPE Summary =====")
print("z0\tz1\tmultiplier\tMAPE_total\tMAPE_g0\tMAPE_g1\tRuntime[s]\tMemory[GB]")
for z0, z1, mult, mt, m0, m1, t, mem in results:
    print(f"{z0}\t{z1}\t{mult}\t{mt:.2e}\t{m0:.2e}\t{m1:.2e}\t{t:.1f}\t{mem:.2f}")
