"""
Metrics module for swarm behavior characterization.

Contains both original metrics (radii, kinetic energy, spacing) and new metrics
for comprehensive behavior discrimination (polarization, angular momentum,
mixing index, cluster count, order parameters).
"""

import numpy as np
import math
from scipy.spatial.distance import pdist, squareform


# =============================================================================
# ORIGINAL METRICS
# =============================================================================

def _centroid(X, mask):
    if mask.sum() == 0:
        return np.array([np.nan, np.nan])
    return X[mask].mean(axis=0)

def avg_radii(X, mask1, mask2):
    c1 = _centroid(X, mask1)
    c2 = _centroid(X, mask2)
    r1 = np.sqrt(((X[mask1] - c1)**2).sum(axis=1)).mean() if mask1.any() else np.nan
    r2 = np.sqrt(((X[mask2] - c2)**2).sum(axis=1)).mean() if mask2.any() else np.nan
    return float(r1), float(r2)

def kinetic_energy(V):
    # 0.5 * mean(|v|^2) with m=1
    return 0.5 * float((V*V).sum(axis=1).mean())

def _pairwise_mean_distance(Xa, Xb):
    # Mean of pairwise distances between rows of Xa and Xb.
    # Vectorized, but O(n*m). For very large N, consider subsampling.
    diff = Xa[:,None,:] - Xb[None,:,:]  # [na, nb, 2]
    D = np.sqrt((diff*diff).sum(axis=2)) # [na, nb]
    return float(D.mean())

def mean_spacing_same(X, mask):
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return float("nan")
    Xa = X[idx]
    # Exclude self-distances by masking the diagonal
    diff = Xa[:,None,:] - Xa[None,:,:]      # [n,n,2]
    D = np.sqrt((diff*diff).sum(axis=2))    # [n,n]
    n = D.shape[0]
    # Take upper triangle without diagonal
    iu = np.triu_indices(n, k=1)
    return float(D[iu].mean())

def mean_spacing_cross(X, mask1, mask2):
    Xa = X[mask1]; Xb = X[mask2]
    if len(Xa)==0 or len(Xb)==0:
        return float("nan")
    return _pairwise_mean_distance(Xa, Xb)

def update_revolutions(c1, c2, prev_angle, revs):
    # Update revolutions count based on centroid-relative angle unwrapping
    r = c1 - c2
    angle = math.atan2(r[1], r[0])
    # unwrap increment to (-pi, pi]
    d = angle - prev_angle
    while d <= -math.pi: d += 2*math.pi
    while d >   math.pi: d -= 2*math.pi
    revs += d / (2*math.pi)
    return revs, angle


# =============================================================================
# NEW METRICS: Polarization (Φ) - Velocity Alignment
# =============================================================================

def polarization_global(V):
    """
    Global polarization across all particles.

    Measures how aligned particle velocities are.

    Formula: Φ = |Σᵢ vᵢ| / Σᵢ |vᵢ|

    Returns:
        float: Polarization in range [0, 1]
        - Φ ≈ 0: Disordered motion (velocities cancel out)
        - Φ ≈ 1: Perfectly aligned flock (all moving same direction)
    """
    speed_sum = np.linalg.norm(V, axis=1).sum()
    if speed_sum < 1e-8:
        return 0.0
    velocity_sum = np.linalg.norm(V.sum(axis=0))
    return float(velocity_sum / speed_sum)


def polarization_species(V, mask):
    """
    Polarization for a single species.

    Args:
        V: velocities [N, 2]
        mask: boolean mask for species particles

    Returns:
        float: Species polarization [0, 1]
    """
    V_species = V[mask]
    if len(V_species) == 0:
        return 0.0
    speed_sum = np.linalg.norm(V_species, axis=1).sum()
    if speed_sum < 1e-8:
        return 0.0
    velocity_sum = np.linalg.norm(V_species.sum(axis=0))
    return float(velocity_sum / speed_sum)


# =============================================================================
# NEW METRICS: Angular Momentum (L) - Global Rotation
# =============================================================================

def angular_momentum_global(X, V):
    """
    Total angular momentum about center of mass.

    Formula: L = Σᵢ (rᵢ - r_cm) × vᵢ

    Returns:
        float: Angular momentum (positive = CCW, negative = CW)
    """
    cm = X.mean(axis=0)
    r = X - cm  # relative positions
    # 2D cross product: r × v = r_x * v_y - r_y * v_x
    L = (r[:, 0] * V[:, 1] - r[:, 1] * V[:, 0]).sum()
    return float(L)


def angular_momentum_species(X, V, mask):
    """
    Angular momentum for a single species about its own centroid.

    Args:
        X: positions [N, 2]
        V: velocities [N, 2]
        mask: boolean mask for species particles

    Returns:
        float: Species angular momentum
    """
    X_s, V_s = X[mask], V[mask]
    if len(X_s) == 0:
        return 0.0
    cm = X_s.mean(axis=0)
    r = X_s - cm
    L = (r[:, 0] * V_s[:, 1] - r[:, 1] * V_s[:, 0]).sum()
    return float(L)


def angular_momentum_normalized(X, V):
    """
    Normalized angular momentum (independent of particle count and speed).

    Normalized by N * mean_radius * mean_speed.

    Returns:
        float: Normalized angular momentum (roughly [-1, 1] for pure rotation)
    """
    cm = X.mean(axis=0)
    r = X - cm
    L = (r[:, 0] * V[:, 1] - r[:, 1] * V[:, 0]).sum()
    # Normalize by N * mean_radius * mean_speed
    mean_r = np.linalg.norm(r, axis=1).mean()
    mean_v = np.linalg.norm(V, axis=1).mean()
    n = len(X)
    norm = n * mean_r * mean_v
    if norm < 1e-8:
        return 0.0
    return float(L / norm)


# =============================================================================
# NEW METRICS: Mixing Index (M) - Species Intermixing
# =============================================================================

def mixing_index(X, species, n_species, r_mix=50.0):
    """
    Compute mixing index based on local species entropy.

    For each particle, count neighbors within radius r_mix and compute
    entropy of species distribution in neighborhood.

    Args:
        X: positions [N, 2]
        species: species labels [N]
        n_species: number of species
        r_mix: neighborhood radius for mixing calculation

    Returns:
        float: Mean local entropy (0 = segregated, log(n_species) = fully mixed)
    """
    n = len(X)
    entropies = []

    for i in range(n):
        # Find neighbors within r_mix
        dist = np.linalg.norm(X - X[i], axis=1)
        neighbors = np.where((dist < r_mix) & (dist > 0))[0]

        if len(neighbors) == 0:
            continue

        # Count species in neighborhood
        neighbor_species = species[neighbors]
        counts = np.bincount(neighbor_species.astype(int), minlength=n_species)
        probs = counts / counts.sum()

        # Entropy (avoid log(0))
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)

    if len(entropies) == 0:
        return 0.0
    return float(np.mean(entropies))


def mixing_index_normalized(X, species, n_species, r_mix=50.0):
    """
    Normalized mixing index [0, 1].

    Args:
        X: positions [N, 2]
        species: species labels [N]
        n_species: number of species
        r_mix: neighborhood radius

    Returns:
        float: Normalized mixing (0 = segregated, 1 = fully mixed)
    """
    M = mixing_index(X, species, n_species, r_mix)
    max_entropy = np.log(n_species)  # Fully mixed case
    if max_entropy < 1e-8:
        return 0.0
    return float(M / max_entropy)


# =============================================================================
# NEW METRICS: Cluster Count (C) - Number of Distinct Groups
# =============================================================================

def cluster_count_simple(X, threshold=50.0):
    """
    Simple cluster count using connected components.
    Two particles are connected if distance < threshold.

    Args:
        X: positions [N, 2]
        threshold: maximum distance for connection

    Returns:
        int: Number of clusters
    """
    n = len(X)
    if n == 0:
        return 0

    # Build adjacency based on distance
    dist = squareform(pdist(X))
    adj = (dist < threshold).astype(int)
    np.fill_diagonal(adj, 0)

    # Find connected components using BFS
    visited = np.zeros(n, dtype=bool)
    n_clusters = 0

    for i in range(n):
        if visited[i]:
            continue
        # BFS from i
        queue = [i]
        visited[i] = True
        while queue:
            node = queue.pop(0)
            for j in np.where(adj[node] > 0)[0]:
                if not visited[j]:
                    visited[j] = True
                    queue.append(j)
        n_clusters += 1

    return n_clusters


def cluster_count_dbscan(X, eps=30.0, min_samples=3):
    """
    Count clusters using DBSCAN.

    Args:
        X: positions [N, 2]
        eps: maximum distance between neighbors in a cluster
        min_samples: minimum points to form a cluster

    Returns:
        int: Number of clusters (excluding noise)
    """
    try:
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        return n_clusters
    except ImportError:
        # Fall back to simple method if sklearn not available
        return cluster_count_simple(X, threshold=eps)


# =============================================================================
# NEW METRICS: Orientation Order Parameter (Ψ) - Angular Alignment
# =============================================================================

def orientation_order(orientations):
    """
    Polar order parameter for orientations.

    Formula: Ψ = |Σᵢ exp(i·θᵢ)| / N

    Args:
        orientations: array of angles [N]

    Returns:
        float: Order parameter [0, 1]
        - Ψ ≈ 0: Random orientations
        - Ψ ≈ 1: All particles pointing same direction
    """
    n = len(orientations)
    if n == 0:
        return 0.0
    # Complex representation: exp(i*theta)
    complex_sum = np.sum(np.exp(1j * orientations))
    return float(np.abs(complex_sum) / n)


def orientation_order_species(orientations, mask):
    """
    Orientation order for a single species.

    Args:
        orientations: array of angles [N]
        mask: boolean mask for species particles

    Returns:
        float: Species orientation order [0, 1]
    """
    return orientation_order(orientations[mask])


def orientation_from_velocity(V):
    """
    Compute orientations from velocity vectors.

    Args:
        V: velocities [N, 2]

    Returns:
        ndarray: orientations [N] in radians
    """
    return np.arctan2(V[:, 1], V[:, 0])


# =============================================================================
# NEW METRICS: Nematic Order Parameter (S) - Alignment Regardless of Direction
# =============================================================================

def nematic_order(orientations):
    """
    Nematic order parameter (head-tail symmetric).

    Measures alignment ignoring head-tail distinction (particles aligned but
    possibly pointing opposite ways).

    Formula: Uses 2*theta to make it 180° periodic

    Args:
        orientations: array of angles [N]

    Returns:
        float: Nematic order [0, 1]
        - S ≈ 0: Random orientations
        - S ≈ 1: All aligned (possibly opposite directions)
    """
    n = len(orientations)
    if n == 0:
        return 0.0
    # Use 2*theta to make it nematic (180° periodic)
    complex_sum = np.sum(np.exp(2j * orientations))
    return float(np.abs(complex_sum) / n)


def nematic_order_species(orientations, mask):
    """
    Nematic order for a single species.

    Args:
        orientations: array of angles [N]
        mask: boolean mask for species particles

    Returns:
        float: Species nematic order [0, 1]
    """
    return nematic_order(orientations[mask])


# =============================================================================
# NEW METRICS: Mean Angular Velocity (ω) - Rotation Speed
# =============================================================================

def mean_angular_velocity(angular_velocities):
    """
    Mean angular velocity magnitude.

    Args:
        angular_velocities: array of angular velocities [N]

    Returns:
        float: Mean |ω|
    """
    return float(np.mean(np.abs(angular_velocities)))


def mean_angular_velocity_signed(angular_velocities):
    """
    Mean angular velocity (signed, shows net rotation direction).

    Args:
        angular_velocities: array of angular velocities [N]

    Returns:
        float: Mean ω (positive = CCW, negative = CW)
    """
    return float(np.mean(angular_velocities))


def angular_velocity_variance(angular_velocities):
    """
    Variance in angular velocities.

    Args:
        angular_velocities: array of angular velocities [N]

    Returns:
        float: Var(ω)
    """
    return float(np.var(angular_velocities))


# =============================================================================
# NEW METRICS: Centroid Velocity (v_cm) - Group Movement Speed
# =============================================================================

def centroid_velocity(V):
    """
    Velocity magnitude of the center of mass.

    Args:
        V: velocities [N, 2]

    Returns:
        float: |v_cm|
    """
    v_cm = V.mean(axis=0)
    return float(np.linalg.norm(v_cm))


def centroid_velocity_species(V, mask):
    """
    Centroid velocity for a single species.

    Args:
        V: velocities [N, 2]
        mask: boolean mask for species particles

    Returns:
        float: |v_cm| for species
    """
    V_s = V[mask]
    if len(V_s) == 0:
        return 0.0
    v_cm = V_s.mean(axis=0)
    return float(np.linalg.norm(v_cm))


def centroid_velocity_vector(V):
    """
    Velocity vector of the center of mass.

    Args:
        V: velocities [N, 2]

    Returns:
        ndarray: v_cm [2]
    """
    return V.mean(axis=0)


# =============================================================================
# CONVENIENCE FUNCTION: Compute All Metrics
# =============================================================================

def compute_all_metrics(X, V, species, n_species, orientations=None, angular_velocities=None,
                        r_mix=50.0, cluster_threshold=50.0):
    """
    Compute all metrics for a simulation snapshot.

    Args:
        X: positions [N, 2]
        V: velocities [N, 2]
        species: species labels [N]
        n_species: number of species
        orientations: particle orientations [N] (optional, derived from V if None)
        angular_velocities: particle angular velocities [N] (optional)
        r_mix: neighborhood radius for mixing calculation
        cluster_threshold: distance threshold for cluster detection

    Returns:
        dict: Dictionary of all computed metrics
    """
    # Create species masks
    masks = [species == i for i in range(n_species)]

    # Derive orientations from velocities if not provided
    if orientations is None:
        orientations = orientation_from_velocity(V)

    metrics = {}

    # Original metrics
    if n_species >= 2:
        r1, r2 = avg_radii(X, masks[0], masks[1])
        metrics['R1'] = r1
        metrics['R2'] = r2
        metrics['Rdiff'] = abs(r1 - r2) if not (np.isnan(r1) or np.isnan(r2)) else np.nan
        metrics['d12'] = mean_spacing_cross(X, masks[0], masks[1])

    metrics['K'] = kinetic_energy(V)

    for i, mask in enumerate(masks):
        metrics[f'd{i+1}{i+1}'] = mean_spacing_same(X, mask)

    # Polarization
    metrics['phi'] = polarization_global(V)
    for i, mask in enumerate(masks):
        metrics[f'phi{i+1}'] = polarization_species(V, mask)

    # Angular momentum
    metrics['L'] = angular_momentum_global(X, V)
    metrics['L_norm'] = angular_momentum_normalized(X, V)
    for i, mask in enumerate(masks):
        metrics[f'L{i+1}'] = angular_momentum_species(X, V, mask)

    # Mixing index
    metrics['M'] = mixing_index(X, species, n_species, r_mix)
    metrics['M_norm'] = mixing_index_normalized(X, species, n_species, r_mix)

    # Cluster count
    metrics['C'] = cluster_count_simple(X, cluster_threshold)

    # Orientation order
    metrics['psi'] = orientation_order(orientations)
    for i, mask in enumerate(masks):
        metrics[f'psi{i+1}'] = orientation_order_species(orientations, mask)

    # Nematic order
    metrics['S'] = nematic_order(orientations)

    # Angular velocity stats (if provided)
    if angular_velocities is not None:
        metrics['omega_mean'] = mean_angular_velocity(angular_velocities)
        metrics['omega_signed'] = mean_angular_velocity_signed(angular_velocities)
        metrics['omega_var'] = angular_velocity_variance(angular_velocities)

    # Centroid velocity
    metrics['v_cm'] = centroid_velocity(V)
    for i, mask in enumerate(masks):
        metrics[f'v_cm{i+1}'] = centroid_velocity_species(V, mask)

    return metrics
