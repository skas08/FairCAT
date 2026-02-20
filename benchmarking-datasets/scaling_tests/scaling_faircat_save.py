import os, time, sys
sys.path.append(os.path.abspath(os.path.join('..')))
import faircat
import numpy as np
import psutil
import scipy.sparse as sp

# user-defined parameters
dist_type_0="powerlaw"
dist_type_1="powerlaw"
k=2
d=2
corr_targets = {}

# M: kxk (moderate homophily)
M = np.array([
    [0.6, 0.1],
    [0.1, 0.6],

])

# D: kxk
D = np.array([
    [0.2, 0.1],
    [0.1, 0.2]
])

# H: (d-1) x k - moderate separation
H =np.array([
    [0.8, 0.2]
    ])

# Pcg: g x k 
g = 2; k = 2 #binary sensitive attribute
Pcg = np.array([[0.7,0.3], [0.3,0.7]]) 


# Define exponent pairs for large-scale test
z_pairs = [(20,20)]
#z_pairs = [(23,23)]
results = []

process = psutil.Process() 

for z_0, z_1 in z_pairs:
    print(f"\n=== Running z0={z_0}, z1={z_1} ===")

    # Compute degree targets 
    deg_0 = 2**(z_0)
    deg_1 = 2**(z_1)

    # Compute node counts per group
    n_0 = int(deg_0 / 64)
    n_1 = int(deg_1 / 64)

    # Compute average and max degree
    avg_deg_0 = deg_0 / n_0
    avg_deg_1 = deg_1 / n_1
    max_deg_0 = int(min(avg_deg_0 * 8, n_0))
    max_deg_1 = int(min(avg_deg_1 * 8, n_1))

    # Measure memory before
    mem_before = process.memory_info().rss / (1024**3) 
    start = time.time()

    # Run FairCAT
    A, X, Label, theta, sorted_attr_group = faircat.faircat(
        n_0, n_1, deg_0, deg_1,
        k, d, max_deg_0, max_deg_1,
        dist_type_0, dist_type_1,
        Pcg, M, D, H,
        att_type="normal",
        corr_targets=corr_targets,
        MAPE=True
    )

    # Measure memory after
    mem_after = process.memory_info().rss / (1024**3)
    mem_used = mem_after - mem_before

    end = time.time()
    runtime = end - start

    adj = A.tocoo() if sp.issparse(A) else sp.coo_matrix(A)

    mask = adj.row < adj.col  # keep one direction
    edges = np.vstack((adj.row[mask], adj.col[mask])).astype(np.int64)

    out_dir = r"data_scaling\faircat_scaling_20"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "edges.npy"), edges)
    np.save(os.path.join(out_dir, "features.npy"), X)
    np.save(os.path.join(out_dir, "labels.npy"),   Label)
    np.save(os.path.join(out_dir, "sens.npy"),   sorted_attr_group)

    # Record results
    results.append((z_0, z_1, runtime, mem_used))
    print(f"Time: {runtime:.1f}s | Memory: {mem_used:.2f} GB")

# Print summary table
print("\n===== Runtime + Memory Summary =====")
print("z0\tz1\tRuntime[s]\tMemory[GB]")
for z0, z1, t, mem in results:
    print(f"{z0}\t{z1}\t{t:.1f}\t{mem:.2f}")
