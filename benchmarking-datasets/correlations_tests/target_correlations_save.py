import os, time, sys
sys.path.append(os.path.abspath(os.path.join('..')))
import faircat
import numpy as np
import numpy as np
import scipy.sparse as sp

# user-defined parameters
k=2
d=2
dist_type_0="powerlaw"
dist_type_1="powerlaw"

# M: kxk (moderate homophily)
M = np.array([
    [0.4, 0.15],
    [0.15, 0.4],

])

# D: kxk
D = np.array([
    [0.20, 0.05],
    [0.05, 0.20],

])

# H: (d-1) x k - moderate separation
H =np.array([
    [0.8, 0.2]
    ])

# Pcg: g x k that does not change class preferences (from M)
g = 2; k = 2 #binary sensitive attribute
Pcg = np.array([[0.7,0.3], [0.3,0.7]]) 

# define the (deg0, deg1) exponent pairs
z_pairs = [(20, 20)]
correlation_configs = [
    #("low", {1: 0.05}),
    #("medium", {1: 0.5})
    ("high", {1: 0.95})
]
    
results = []

for z_0, z_1 in z_pairs:
    for label, corr in correlation_configs:
        print(f"\n=== Running z0={z_0}, z1={z_1} | config={label} ===")

        # Compute group-specific degree targets
        deg_0 =  (2**z_0)
        deg_1 =  (2**z_1)

        # Compute node counts per group
        n_0 = int(deg_0 / 32)
        n_1 = int(deg_1 / 32)

        # Compute average and max degree
        avg_deg_0 = deg_0 / n_0
        avg_deg_1 = deg_1 / n_1
        max_deg_0 = int(min(avg_deg_0 * 8, n_0))
        max_deg_1 = int(min(avg_deg_1 * 8, n_1))

        start = time.time()

        corr_targets = corr
        # Run FairCAT
        S, X, Label, theta, sorted_attr_group = faircat.faircat(
            n_0, n_1, deg_0, deg_1,
            k, d, max_deg_0, max_deg_1,
            dist_type_0, dist_type_1,
            Pcg, M, D, H,
            att_type="normal",
            corr_targets=corr_targets,
            MAPE=True
        )

        # Compute mape
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

        g = np.asarray(sorted_attr_group)
        mape_0 = np.mean(np.abs(theta_actual[g==0] - theta_target[g==0]) / theta_target[g==0])
        mape_1 = np.mean(np.abs(theta_actual[g==1] - theta_target[g==1]) / theta_target[g==1])

                # achieved correlation 
        s = X[:, 0].astype(float)        # sensitive attribute (column 0)
        x_target = X[:, 1].astype(float) # non-sensitive attribute

        # actual correlation in the generated data
        corr_actual = np.corrcoef(s, x_target)[0, 1]

        #  target correlation from corr_targets
        target_corr = list(corr_targets.values())[0]

        print(f"Target corr={target_corr:.2f} | Actual corr={corr_actual:.3f} | Diff={abs(corr_actual - target_corr):.3e}")

        end = time.time()
        runtime = end - start

        adj = S.tocoo() if sp.issparse(S) else sp.coo_matrix(S)


        mask = adj.row < adj.col  # keep one direction
        edges = np.vstack((adj.row[mask], adj.col[mask])).astype(np.int64)

        out_dir = r"data_scaling\faircat_correlations_high"
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "edges.npy"), edges)
        np.save(os.path.join(out_dir, "features.npy"), X)
        np.save(os.path.join(out_dir, "labels.npy"),   Label)
        np.save(os.path.join(out_dir, "sens.npy"),   sorted_attr_group)

        # Record it
        results.append((z_0, z_1, label, mape_total, mape_0, mape_1, runtime, target_corr, corr_actual))




        print(f"MAPE overall: {mape_total:.2e} | g0: {mape_0:.2e} | g1: {mape_1:.2e} | Time: {runtime:.1f}s")

print("\n===== Summary =====")
print("z0\tz1\tconfig\tMAPE_total\tMAPE_g0\tMAPE_g1\tTarget_corr\tActual_corr\tRuntime[s]")
for z0, z1, cfg, mt, m0, m1, t, tc, ac in results:
    print(f"{z0}\t{z1}\t{cfg}\t{mt:.2e}\t{m0:.2e}\t{m1:.2e}\t{tc:.2f}\t{ac:.3f}\t{t:.1f}")

