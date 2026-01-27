import os, sys
sys.path.append(os.path.abspath(os.path.join('..')))
import faircat
import numpy as np
import time, psutil

# user-defined parameters
dist_type_0="powerlaw"
dist_type_1="powerlaw"
k=5
d=2
corr_targets = {}

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
H =np.array([
    [0.4, 0.1, 0.1, 0.4, 0.1]
    ])

# Pcg
g = 2
base = np.array([0.4, 0.15, 0.15, 0.15, 0.15]) 
Pcg = np.vstack([base, base])     
Pcg /= Pcg.sum(axis=1, keepdims=True)


z_pairs = [(11,11), (12,12)]
# z_pairs = [(15,15)]
results = []

process = psutil.Process()  

for z_0, z_1 in z_pairs:
    print(f"\n=== Running z0={z_0}, z1={z_1} ===")


    deg_0 = 2**(z_0)
    deg_1 = 2**(z_1)


    n_0 = int(deg_0 / 64)
    n_1 = int(deg_1 / 64)

    avg_deg_0 = deg_0 / n_0
    avg_deg_1 = deg_1 / n_1
    max_deg_0 = int(min(avg_deg_0 * 8, n_0))
    max_deg_1 = int(min(avg_deg_1 * 8, n_1))

    mem_before = process.memory_info().rss / (1024**3)  # GB
    start = time.time()

    # faircat run
    S, X, Label= faircat.faircat(
        n_0, n_1, deg_0, deg_1,
        k, d, max_deg_0, max_deg_1,
        dist_type_0, dist_type_1,
        Pcg, M, D, H,
        att_type="normal",
        corr_targets=corr_targets
    )

    # mem after
    mem_after = process.memory_info().rss / (1024**3)
    mem_used = mem_after - mem_before


    end = time.time()
    runtime = end - start


    # results
    results.append((z_0, z_1, runtime, mem_used))
    print(f"Time: {runtime:.1f}s | Memory: {mem_used:.2f} GB")

print("\n===== Runtime + Memory Summary =====")
print("z0\tz1\tRuntime[s]\tMemory[GB]")
for z0, z1, t, mem in results:
    print(f"{z0}\t{z1}\t{t:.1f}\t{mem:.2f}")
