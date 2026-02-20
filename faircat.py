import numpy as np
from numpy import linalg as la
from scipy import sparse
from scipy.stats import bernoulli
import random
import copy
import powerlaw
import warnings
warnings.simplefilter('ignore')


def node_deg(n_0, n_1, deg_0, deg_1, max_deg_0, max_deg_1, dist_type_0, dist_type_1, norm_sigma_0=None, norm_sigma_1=None):
    """
    generate separate degree distributions for binary sensitive attribute groups.
    input:
    - n_0, n_1: number of nodes in group 0 and 1
    - m_0, m_1: target number of degrees for group 0 and 1
    - max_deg_0, max_deg_1: maximum degrees for nodes in each group
    - dist_type_0, dist_type_1: distribution type for group 0 and 1 ('powerlaw', 'uniform', 'normal')

    output:
    - theta_0: list of degrees for group 0
    - theta_1: list of degrees for group 1
    """


    def generate_degrees(n, deg_count, max_deg, dist_type, norm_sigma=None):
        p = 3.

        if dist_type == 'powerlaw':
            simulated_data = [0]
            while sum(simulated_data) < deg_count:
                theoretical_distribution = powerlaw.Power_Law(xmin = 1., parameters = [p])
                simulated_data=theoretical_distribution.generate_random(n)
                over_list = np.where(simulated_data>max_deg)[0]
                while len(over_list) != 0:
                    add_deg = theoretical_distribution.generate_random(len(over_list))
                    for i,node_id in enumerate(over_list):
                        simulated_data[node_id] = add_deg[i]
                    over_list = np.where(simulated_data>max_deg)[0]
                simulated_data = np.round(simulated_data)
                if (deg_count - sum(simulated_data)/2) < deg_count/5: 
                    p -= 0.01 # small decrease of p
                else:
                    p -= 0.1 # bigger decrease of p
                if p<1.01:
                    print("break")
                    break
        
        elif dist_type == 'uniform': 
            target_mean = deg_count / n
            # range of variation around the mean (cannot go lower than 1 or higher than max_deg)
            range_width = min(target_mean - 1, max_deg - target_mean) # half-width of the uniform distribution
            low = max(1, target_mean - range_width) # lower of the two ranges
            high = min(max_deg, target_mean + range_width)
            simulated_data = np.random.uniform(low, high, size=n)
            simulated_data = np.round(np.clip(simulated_data, 1, max_deg)) #ensure values within bounds

        elif dist_type == 'normal':
            mu = deg_count / n #  mean
            sigma_used = norm_sigma if norm_sigma is not None else mu / 2
            simulated_data = np.random.normal(mu, sigma_used, size=n)
            simulated_data = np.round(np.clip(simulated_data, 1, max_deg))


        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        print(f"sum of degrees for group (target={deg_count}):", int(sum(simulated_data)))
        return simulated_data

    theta_0 = generate_degrees(n_0, deg_0, max_deg_0, dist_type_0, norm_sigma=norm_sigma_0)
    theta_1 = generate_degrees(n_1, deg_1, max_deg_1, dist_type_1, norm_sigma=norm_sigma_1)

    return theta_0, theta_1

def prepare_theta_and_indices(theta_0, theta_1):
    """
    concatenate and sort degrees from two attribute groups

    Parameters:
    - theta_0: degrees for attribute group 0
    - theta_1: degrees for attribute group 1

    returns:
        - sorted_theta: degrees in descending order
        - sorted_attr_group: attribute group (0 or 1) of each sorted node
    """
    theta_combined = np.array(list(theta_0) + list(theta_1))
    attr_labels = [0] * len(theta_0) + [1] * len(theta_1)
    sorted_indices = np.argsort(-theta_combined) # O(n log n) sort in descending order
    sorted_theta = theta_combined[sorted_indices]
    sorted_attr_group = np.array(attr_labels)[sorted_indices]
    
    return sorted_theta.tolist(), sorted_attr_group.tolist()



def count_edges(A):
    """
    Counts the edges

    Parameters
    - A: adjacency matrix (n x n)

    Returns
    - total number of edges in the graph
    """
    n = A.shape[0]
    node_degree = np.zeros(n)
    nnz = A.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            node_degree[nnz[0][i]] += 1
            node_degree[nnz[1][i]] += 1
    return int(sum(node_degree)/2)

    
def generate_classes_from_group(k, groups, Pcg):
    """
    Generates class labels for each node based on its sensitive group membership and a conditional probability matrix P(class | group).

    Parameters:
    - k: number of classes
    - groups: list or array of group labels (length n)
    - Pcg: array of shape (num_groups, num_classes)

    Returns:
    - C: list of class labels (length n)
    """
    C = [
        np.random.choice(k, p=Pcg[g])
        for g in groups
    ]
    return C

def latent_factor_gen_from_C(C,n,k,M,D):
    """
    generating latent factors U based on class labels and user-defined M and D 
    
    input:
    - C: class labels for each node 

    returns:
    - U: latent factors for each node
    """
    density = np.zeros(k)
    for l in range(k):
        density[l] = M[l,l]


    U = np.zeros((n,k))

    for i in range(n):
        class_label = C[i]
        for h in range(k):
            U[i, h] = np.random.normal(loc=M[class_label][h], scale=D[class_label][h])

    # eliminate U<0 and U>1 (keep 0<=U<=1)
    minus_list = np.where(U < 0)
    for i in range(len(minus_list[0])):
        U[minus_list[0][i],minus_list[1][i]] = 0
    one_list = np.where(U > 1)
    for i in range(len(one_list[0])):
        U[one_list[0][i],one_list[1][i]] = 1
    # normalize with guard against zero/NaN rows
    for i in range(n):
        row_sum = np.sum(U[i])
        if row_sum > 0 and np.isfinite(row_sum):
            U[i] /= row_sum
        else:
            U[i] = np.ones(k) / k

    return U,density


def adjust(n,k,U,C,M):
    """
    taken entirely from GenCAT
    adjusting latent factors U based on class-wise edge density M
    """
    U_prime = copy.deepcopy(U)
    partition = []
    for l in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)
        
    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))
    
    def inverse(U_tmp,l):
        U_ = 1 - U_tmp
        sum_U_ = sum(U_) - U_tmp[l]
        for i in range(k):
            if i != l:
                U_[i] = U_[i] * U_tmp[l] / sum_U_
        return U_
    flag=0
    for l in range(k):
        loss_min = float('inf')
        if  M[l][l] >= 1/k:
            for Th in np.arange(0.01,1,0.05):
                sum_estimated = np.zeros(k)
                for i in partition[l]:
                    sum_estimated += freez_func(U[i],Th) * freez_func(U[i],Th)
                loss_tmp = la.norm(M[l]-sum_estimated/len(partition[l]))
                if loss_tmp < loss_min:
                    loss_min = loss_tmp
                    Th_min = Th
            for i in partition[l]:
                U[i] = freez_func(U[i],Th_min)
                U_prime[i] = U[i]
        else:
            for Th in np.arange(0.01,1,0.05):
                sum_estimated = np.zeros(k)
                for i in partition[l]:
                    sum_estimated += freez_func(U[i],Th) * inverse(freez_func(U[i],Th),l)
                loss_tmp = la.norm(M[l]-sum_estimated/len(partition[l]))
                if loss_tmp < loss_min:
                    loss_min = loss_tmp
                    Th_min = Th
            for i in partition[l]:
                U[i] = freez_func(U[i],Th_min)
                U_prime[i] = inverse(U[i],l)
    return U, U_prime
        
def edge_construction(n, U, k, U_prime, step, theta, r):
    """
    taken entirely from GenCAT

    generating edges based on latent factors U and U_prime and degree list theta
    """
    U_ = copy.deepcopy(U)
    
    A = sparse.dok_matrix((n,n))
    degree_list = np.zeros(n)
    count_list = []
    print_count = 1
    for i in range(n):
        count = 0
        ng_list = set([i])
        while count < r and degree_list[i] < theta[i]:
            to_classes = random.choices(list(range(0,k)), k=int(theta[i]-degree_list[i]), weights=U_[i])
            for to_class in to_classes:
                for loop in range(50):
                    j = U_prime[to_class][int(random.random()/step)]
                    if j not in ng_list:
                        ng_list.add(j)
                        break
                if degree_list[j] < theta[j] and i!=j:
                    A[i,j] = 1;A[j,i] = 1
                    degree_list[i]+=1;degree_list[j]+=1
            count += 1 
        count_list.append(count)
    return A, count_list

def ITS_U_prime(n,k,U_prime,step):
    """
    taken entirely from GenCAT
    Inverse Transform Sampling for U_prime
    """
    class_list = []
    UT = U_prime.transpose()
    for i in range(k): 
        UT_tmp = UT[i]/ sum(UT[i])
        for j in range(n-1):
            UT_tmp[j+1] += UT_tmp[j]

        class_tmp = []
        node_counter = 0
        for l in np.arange(0,1,step):
            if node_counter >= n-1:
                class_tmp.append(n-1)
            elif UT_tmp[node_counter] > l:
                class_tmp.append(node_counter)
            else:
                node_counter += 1
                class_tmp.append(node_counter)
        class_list.append(class_tmp)
    return class_list


def adjust_att(n,k,d,U,C,H):
    """
    taken entirely from GenCAT
    adjusting attribute latent factors V 
    """
    V = copy.deepcopy(H)
    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)
        
    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))
    
    P = np.zeros((k,k))
    for l in range(k):
        for j in partition[l]:
            P[l] += U[j]
        P[l] = P[l]/len(partition[l])
        
    for delta in range(d):
        loss = []
        for Th in np.arange(0.1, 1.1, 0.05):
            loss.append(np.linalg.norm(H[delta] - P @ freez_func(V[delta],Th).T))
        V[delta] = freez_func(V[delta],0.1*(np.argmin(loss)+1))
    return V

def attribute_generation(n,d,k,U,V,C,omega,att_type):
    """
    taken from GenCAT
    generating attributes X based on latent factors U and V
    """
    X = U@V.T
    X_keep = copy.deepcopy(X) # keep original probabilities for Bernoulli adjustment later
    def variation_attribute(n,d,k,X,C,att_type):
        if att_type == "normal":
            for i in range(d): # each attribute demension
                clus_dev = np.random.uniform(omega,omega,k) # variation for each class
                for p in range(n): # each node
                    X[p,i] += np.random.normal(0.0,clus_dev[C[p]],1)
            # normalization
            for i in range(d):
                X[:,i] -= min(X[:,i])
                X[:,i] /= max(X[:,i])
        else: # Bernoulli distribution
            for i in range(d):
                for p in range(n):
                    X[p,i] = bernoulli.rvs(p=X[p,i], size=1)   
            
        return X
    return variation_attribute(n,d,k,X,C,att_type), X_keep

####### BINARY–CONTINUOUS CORRELATION ADJUSTMENT ##############

def _solve_alpha_for_target_corr(x, s, phi_star): 
    """
    Find alpha such that corr(x', s) = phi_star, where x' = x + alpha*sc
    
    Parameters:
    - x: original continuous non-sensitive attribute values
    - s: sensitive attribute values
    - phi_star: target correlation

    Returns:
    - alpha: the value to adjust x by
    """
    x = np.asarray(x, float).ravel(); s = np.asarray(s, float).ravel() #transform it into float and make it one-dimensional
    xc = x - x.mean(); sc = s - s.mean() #centered
    var_x = np.dot(xc, xc) # variance of x
    var_s = np.dot(sc, sc) # variance of s
    cov_xs = np.dot(xc, sc) # covariance of x and s

    # if all values in s are identical
    if var_s <= 0:
        return 0.0

    # quadratic formula: A*alpha^2 + B*alpha + C = 0

    A = (phi_star**2-1) * (var_s**2)
    B = 2 * (phi_star**2-1) * cov_xs * var_s
    C = (phi_star**2) * var_x * var_s - cov_xs**2

    roots = np.roots([A, B, C]) if abs(A) > 1e-18 else [-C/B]  # to not divide by zero
    # Choose the root that yields the closer correlation
    best_alpha = None; best_err = float("inf")
    for a in roots:
        if not np.isreal(a): #complex numbers don't make sense here
            continue
        a = float(np.real(a))
        # evaluate corr for candidate
        num = cov_xs + a * var_s
        den = np.sqrt((var_x + 2*a*cov_xs + a*a*var_s) * var_s) #denominator
        if den <= 0: # avoid division by zero and invalid values
            continue
        phi = num / den 
        err = abs(phi - phi_star)
        if err < best_err:
            best_err = err
            best_alpha = a
    return 0.0 if best_alpha is None else best_alpha

def adjust_feature_to_corr_cont(x, s, r_target):
    """
    Adjust continuous attribute x to have target correlation r_target with sensitive attribute s.
    
    inputs:
    - x: continuous non-sensitive attribute values 
    - s: sensitive attribute values 
    - r_target: target correlation value

    Returns:
    - x' with corr(x', s) ~= r_target. (s is untouched)
    """
    x = np.asarray(x, float).ravel(); s = np.asarray(s, float).ravel()
    xc = x - x.mean(); sc = s - s.mean()

    alpha = _solve_alpha_for_target_corr(x, s, r_target)
    x_new = xc + alpha * sc + x.mean()
    if x_new.min() < 0 or x_new.max() > 1:
        # re-scale to [0,1]
        print("Rescaling adjusted continuous attribute to [0,1]")
        x_new -= x_new.min() # eg. 1.2-(-0.3) = 1.5 or -0.3-(-0.3)=0.0
        if x_new.max() > 0:
            x_new /= x_new.max() # eg. 1.5/1.5 = 1.0
        else:
            x_new = np.zeros_like(x_new) # all values are zero

    return x_new


###### BINARY–BINARY CORRELATION ADJUSTMENT ##############


def _phi_feasible_interval(pi, p):
    """
    Gives feasible interval of phi values

    parameters:
    - pi: P(s=1)
    - p: P(x=1)

    returns:
    - (lo, hi): feasible interval of phi values
    """
    # Delta bounds:
    # Delta  [ max(-(1-p)/pi, -p/(1-pi)),  min(p/pi, (1-p)/(1-pi)) ]
    lo = max(-(1.0 - p)/pi if pi > 0 else -np.inf,
             -p/(1.0 - pi)   if pi < 1 else -np.inf)
    hi = min(p/pi           if pi > 0 else  np.inf,
             (1.0 - p)/(1.0 - pi) if pi < 1 else np.inf)
    # map delta bounds to phi bounds: phi = Delta * sqrt(pi(1-pi)) / sqrt(p(1-p))
    scale = np.sqrt(pi * (1.0 - pi)) / np.sqrt(max(p*(1.0 - p), 1e-18))
    return lo * scale, hi * scale





def flip_min_to_match_rate(x_bin, target_rate, probs_col, idxs):
    """
    Flip bits in x_bin to match the target rate, using the probabilities in probs_col.

    parameters:
    - x_bin: binary array (0/1)
    - target_rate: desired rate of 1s in x_bin (between 0 and 1)
    - probs_col: array of probabilities associated with each element in x_bin
    - idxs: array of original indices corresponding to x_bin

    returns:
    - modified x with rate of 1s equal to target_rate
    """
    x = np.asarray(x_bin, int).ravel().copy()
    p = probs_col
    num = x.size
    if p.size != num:
        print(p.size,num)
        raise ValueError("`probs` must have same length as `x_bin`.")
    p = np.clip(p, 1e-12, 1-1e-12) # to avoid numerical issues

    want_ones = int(round(target_rate * num))
    have_ones = int(x.sum())
    diff = want_ones - have_ones
    if diff == 0:
        return x


    idxs = np.asarray(idxs).ravel()
    if idxs.size != num:
        raise ValueError("`idxs` must have same length as `x_bin`.")


    if diff > 0:
        # flipping 0 to 1s: choose zeros with largest probs
        idx0 = np.flatnonzero(x == 0) #to get indices of x=0
        if idx0.size < diff:
            raise ValueError("Not enough zeros to flip.")
        order = np.argsort(-p[idx0], kind="mergesort")  # largest first
        flip = idx0[order[:diff]] #takes the first diff indices of idx0 after ordering
        global_flip_idx = idxs[flip] #incidences in the original array that are flipped
        x[flip] = 1
    else:
        # flipping 1s to 0s: choose ones with smallest probs
        diff = -diff
        idx1 = np.flatnonzero(x == 1)
        if idx1.size < diff:
            raise ValueError("Not enough ones to flip.")
        order = np.argsort(p[idx1],  kind="mergesort")  # smallest first
        flip = idx1[order[:diff]]
        global_flip_idx = idxs[flip]
        x[flip] = 0

    return x


def adjust_feature_to_corr_phi(x_bin, s_bin, phi_target, probs_col):
    """
    Adjust binary attribute

    parameters:
    - x_bin: binary non-sensitive attribute 
    - s_bin: binary sensitive attribute
    - phi_target: target phi correlation value
    - probs_col: per-node probabilities for x=1

    returns:
    - x' with phi(x', s) ~= phi_target
    """
    xb = np.asarray(x_bin, int).ravel().copy() #non-sensitive attribute
    sb = np.asarray(s_bin, int).ravel() #sensitive attribute
    if set(np.unique(xb)) - {0,1} or set(np.unique(sb)) - {0,1}:
        raise ValueError("Binary path requires {0,1} inputs.")

    pi = sb.mean()        # pi = P(s=1)
    p  = xb.mean()        # p = P(x=1)
    if not (0.0 < pi < 1.0):
        return xb  # nothing to do due to s having no variation

    # Feasibility of requested phi
    lo_phi, hi_phi = _phi_feasible_interval(pi, p)
    if phi_target < lo_phi:
        phi_target = lo_phi
        print(f"Warning: Target phi too low; adjusted to feasible minimum {lo_phi:.3f}.")
    elif phi_target > hi_phi:
        phi_target = hi_phi
        print(f"Warning: Target phi too high; adjusted to feasible maximum {hi_phi:.3f}.")

    # calculate delta (= p1 - p0)
    # delta = phi * sqrt(p(1-p)) / sqrt(pi(1-pi))
    Delta = phi_target * np.sqrt(max(p*(1.0 - p), 1e-18)) / np.sqrt(pi * (1.0 - pi))

    # mixture constraint: p = (1-pi)p0 + pi p1 ⇒ p0 = p - piDelta, p1 = p + (1-pi)Delta
    p0 = p - pi * Delta
    p1 = p + (1.0 - pi) * Delta


    # Flip minimal number of 0/1s within each s group to hit target counts
    x_new = xb.copy()
    g0 = np.flatnonzero(sb == 0)
    g1 = np.flatnonzero(sb == 1)
    x_new[g0] = flip_min_to_match_rate(x_new[g0], p0, probs_col[g0],idxs=g0)
    x_new[g1] = flip_min_to_match_rate(x_new[g1], p1, probs_col[g1],idxs=g1)

    return x_new


def adjust_columns_to_corr(X, s, corr_spec, attr_type,X_keep):
    """
    Adjust specified columns of X to have target correlations with sensitive attribute s. Leave other columns untouched. Works for normal or bernoulli attributes.

    parameters:
    - X: (n x d) attributes after the generate-attributes step
    - s: (n,) sensitive attribute (0/1)
    - corr_spec: list/array of target correlations (length d)
    - attr_type: "bernoulli" or "normal"

    returns:
    - X with some specified columns adjusted to target correlations
    """

    if not isinstance(corr_spec, dict):
        raise TypeError("corr_spec must be a dict mapping column index -> target correlation")

    n, d = X.shape
    s_int = s.astype(int)

    # iterate only over requested columns
    for j, r_tgt in corr_spec.items():
        if not isinstance(j, (int, np.integer)):
            raise KeyError(f"corr_spec key is not an integer column index")
        if not (0 <= j < d):
            raise KeyError(f"corr_spec key {j} out of range 0..{d-1}")

        r_tgt = float(r_tgt)

        if attr_type == "normal":
            # continuous
            X[:, j] = adjust_feature_to_corr_cont(
                X[:, j].astype(float),
                s_int.astype(float),
                r_tgt
            )

        elif attr_type == "bernoulli":
            col_bin = X[:, j].astype(int)

            # group counters
            if np.any(s_int == 0):
                p0 = float(col_bin[s_int == 0].mean())
            else:
                p0 = float(col_bin.mean())
            if np.any(s_int == 1):
                p1 = float(col_bin[s_int == 1].mean())
            else:
                p1 = float(col_bin.mean())

            # map X column j to row in V (attributes are X_rest columns)
            col_index = j - 1  # as j only includes non-sensitive attributes
            probs_col = X_keep[:, col_index].astype(float)
            X[:, j] = adjust_feature_to_corr_phi(
                col_bin, s_bin=s_int, phi_target=r_tgt, probs_col=probs_col
            )
        else:
            raise ValueError("attr_type must be 'normal' or 'bernoulli'.")

    return X


######### main function ########################

def faircat(n_0, n_1,deg_0, deg_1,k,d,max_deg_0, max_deg_1,dist_type_0, dist_type_1, Pcg, M,D,H,phi_c=1,omega=0.2,r=50,step=100,att_type="bernoulli",norm_sigma_0=None, norm_sigma_1=None, corr_targets=False, MAPE=False):
    # node degree generation 
    theta_0, theta_1 = node_deg(
    n_0, n_1, deg_0, deg_1, max_deg_0, max_deg_1,
    dist_type_0, dist_type_1,
    norm_sigma_0=norm_sigma_0, norm_sigma_1=norm_sigma_1)

    theta, sorted_attr_group = prepare_theta_and_indices(theta_0, theta_1)
    n=n_0+n_1


    # class generation
    C = generate_classes_from_group(k, np.array(sorted_attr_group).astype(int), Pcg)
    # latent factor U generation
    U,density = latent_factor_gen_from_C(C, n, k, M, D)
    
    # adjusting phase
    U,U_prime = adjust(n,k,U,C,M)

    # Inverse Transform Sampling
    step = 1/(n*step)
    U_prime_CDF = ITS_U_prime(n,k,U_prime,step)

    # Edge generation
    A_gen, count_list = edge_construction(n, U, k, U_prime_CDF, step, theta, r)
        
    print("number of generated edges : " + str(count_edges(A_gen)))



    # Generate (d - 1) attributes using U and V
    V = adjust_att(n, k, d - 1, U, C, H)  # only d-1 because the first attribute will be group
    X_rest, X_keep = attribute_generation(n, d - 1, k, U, V, C, omega, att_type)

    # Add sensitive label as the first column
    group_column = np.array(sorted_attr_group).reshape(-1, 1)
    X = np.hstack([group_column, X_rest])  # Concatenate as first column

    # Apply correlation adjustments if specified
    if corr_targets is not None and len(corr_targets) > 0:
        corr_targets = {j: r for j, r in corr_targets.items() if j != 0 and 0 <= j < X.shape[1]}
        if len(corr_targets) > 0:
            s = X[:, 0]  # sensitive attribute 
            X = adjust_columns_to_corr(
                X, s, corr_targets,
                attr_type=att_type, X_keep=X_keep  
            )
    if MAPE==True:
        return A_gen,X,C, theta, sorted_attr_group
    else:
        return A_gen,X,C


######### Class reproduction function ########################
# taken from GenCAT
def class_reproduction(k,A,Label):
    # extract class preference matrix from given graph
    pref = np.zeros((len(Label),k))
    nnz = A.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            pref[nnz[0][i]][Label[nnz[1][i]]] += 1
            pref[nnz[1][i]][Label[nnz[0][i]]] += 1
    for i in range(len(Label)):
        pref[i] /= sum(pref[i])

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)
        
    # caluculate average and deviation of class preference
    from statistics import mean, median,variance,stdev
    M = np.zeros((k,k))
    D = np.zeros((k,k))
    for i in range(k):
        pref_tmp = []
        for j in partition[i]:
            pref_tmp.append(pref[j])
        pref_tmp = np.array(pref_tmp).transpose()
        for h in range(k):
            M[i,h] = mean(pref_tmp[h])
            D[i,h] = stdev(pref_tmp[h])
    
    return M,D
