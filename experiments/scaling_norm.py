import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import faircat
import numpy as np
import time, psutil, gc

dist_type_0="powerlaw"
dist_type_1="powerlaw"
k=5
d=(2**10)+1

z_0 = 20
z_1 = 20

# M: kxk (moderate homophily)
M = np.array([
    [0.6, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.6, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.6, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.6, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.6],
])

# D: kxk
D = np.array([
    [0.2, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.2, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.2, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.2, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.2],
])

# H: (d-1) x k - moderate separation
H = np.random.rand(d - 1, k)
H /= H.sum(axis=1, keepdims=True)

# Pcg
g = 2
base = np.array([0.4, 0.15, 0.15, 0.15, 0.15])  # sums to 1 after normalization
Pcg = np.vstack([base, base])  
Pcg /= Pcg.sum(axis=1, keepdims=True)


results = []

process = psutil.Process() 

corr_targets_lengths = [0,2,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10]

for c in corr_targets_lengths:
    print(f"\n=== Running z0={z_0}, z1={z_1}, c={c} ===")

    deg_0 = 2**z_0
    deg_1 = 2**z_1

    n_0 = int(deg_0 / 64)
    n_1 = int(deg_1 / 64)

    avg_deg_0 = deg_0 / n_0
    avg_deg_1 = deg_1 / n_1
    max_deg_0 = int(min(avg_deg_0 * 8, n_0))
    max_deg_1 = int(min(avg_deg_1 * 8, n_1))

    corr_vals = np.random.uniform(-1, 1, size=c) 
    corr_targets = {i+1: v for i, v in enumerate(corr_vals)}
    print(f"Correlation targets: {corr_targets}")

    mem_before = process.memory_info().rss / (1024**3)  
    start = time.time()

    # faircat run
    A, X, Label = faircat.faircat(
        n_0, n_1, deg_0, deg_1,
        k, d, max_deg_0, max_deg_1,
        dist_type_0, dist_type_1,
        Pcg, M, D, H,
        corr_targets=corr_targets,
        att_type="normal"
    )

    mem_after = process.memory_info().rss / (1024**3)
    mem_used = mem_after - mem_before


    end = time.time()
    runtime = end - start

    results.append((c, runtime, mem_used))
    print(f"Time: {runtime:.1f}s | Memory: {mem_used:.2f} GB")


# cleanup
    del A, X, Label
    del corr_vals, corr_targets
    gc.collect()
