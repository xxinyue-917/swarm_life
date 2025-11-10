
import numpy as np
import math

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
