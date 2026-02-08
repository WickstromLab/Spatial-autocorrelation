"""
Spatial Autocorrelation Analysis using Moran's I Index

Analyzes whether objects in microscopy images are spatially clustered,
dispersed, or randomly distributed.

Supported input types:
    - Binary images: Binarized fluorescence microscopy images (use skimage.measure.label
      to convert to label mask before analysis)
    - Label masks: Pre-segmented images where each object has a unique integer ID

Pure NumPy/SciPy implementation - no external geospatial dependencies required.

Dependencies:
    pip install numpy scipy scikit-image matplotlib

Example usage with binarized fluorescence image:
    from skimage import io
    from skimage.measure import label
    from spatial_autocorrelation import analyze_with_random_control

    binary_img = io.imread('binarized_fluorescence.tif')
    label_mask = label(binary_img > 0)
    results = analyze_with_random_control(label_mask, save_path='output.pdf')
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from skimage.measure import regionprops
import matplotlib.pyplot as plt


def extract_object_centroids(label_mask, min_area=0):
    """
    Extract centroid coordinates from a labeled image.

    Parameters
    ----------
    label_mask : ndarray
        2D array where each object has a unique integer ID (0 = background)
        Supports 8-bit, 16-bit, 32-bit, and float images (will be converted to int)
    min_area : int
        Minimum object area in pixels (filters out small objects/noise)

    Returns
    -------
    centroids : ndarray
        Array of shape (n_objects, 2) with (row, col) coordinates
    labels : ndarray
        Array of object labels corresponding to each centroid
    """
    # Convert to integer type if needed (handles 32-bit float, etc.)
    if not np.issubdtype(label_mask.dtype, np.integer):
        label_mask = label_mask.astype(np.int32)

    props = regionprops(label_mask)
    if len(props) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    # Filter by minimum area
    if min_area > 0:
        props = [p for p in props if p.area >= min_area]
        if len(props) == 0:
            return np.array([]).reshape(0, 2), np.array([])

    centroids = np.array([p.centroid for p in props])
    labels = np.array([p.label for p in props])
    return centroids, labels


def build_spatial_weights_knn(coordinates, k=8):
    """
    Build a K-nearest neighbors spatial weights matrix.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape (n, 2) with coordinates
    k : int
        Number of nearest neighbors

    Returns
    -------
    W : ndarray
        Row-standardized spatial weights matrix (n x n)
    """
    n = len(coordinates)
    if n < 2:
        raise ValueError("Need at least 2 points")

    k = min(k, n - 1)
    distances = cdist(coordinates, coordinates)

    W = np.zeros((n, n))
    for i in range(n):
        # Get indices of k nearest neighbors (excluding self)
        dist_i = distances[i].copy()
        dist_i[i] = np.inf  # Exclude self
        neighbors = np.argsort(dist_i)[:k]
        W[i, neighbors] = 1

    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums

    return W


def build_spatial_weights_distance(coordinates, threshold=None):
    """
    Build a distance-based spatial weights matrix.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape (n, 2) with coordinates
    threshold : float
        Distance threshold. If None, uses 10% of extent diagonal.

    Returns
    -------
    W : ndarray
        Row-standardized spatial weights matrix (n x n)
    """
    n = len(coordinates)
    if n < 2:
        raise ValueError("Need at least 2 points")

    distances = cdist(coordinates, coordinates)

    if threshold is None:
        extent = np.ptp(coordinates, axis=0)
        threshold = np.sqrt(np.sum(extent**2)) * 0.1

    W = (distances <= threshold).astype(float)
    np.fill_diagonal(W, 0)  # No self-neighbors

    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    return W


def build_spatial_weights_delaunay(coordinates, distance_threshold=None):
    """
    Build spatial weights based on Delaunay triangulation with optional distance threshold.

    Delaunay triangulation connects points such that no point is inside the circumcircle
    of any triangle. This creates a natural neighbor structure. The optional distance
    threshold removes edges longer than the specified distance.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape (n, 2) with coordinates
    distance_threshold : float, optional
        Maximum edge length to retain. If None, all Delaunay edges are kept.
        If provided, edges longer than this threshold are removed.

    Returns
    -------
    W : ndarray
        Row-standardized spatial weights matrix (n x n)
    edges : list
        List of (i, j) tuples representing the edges (for visualization)
    """
    n = len(coordinates)
    if n < 3:
        raise ValueError("Need at least 3 points for Delaunay triangulation")

    # Compute Delaunay triangulation
    tri = Delaunay(coordinates)

    # Extract edges from triangulation
    W = np.zeros((n, n))
    edges = set()

    for simplex in tri.simplices:
        # Each simplex (triangle) has 3 vertices - add all edges
        for i in range(3):
            for j in range(i + 1, 3):
                v1, v2 = simplex[i], simplex[j]

                # Calculate edge length
                dist = np.sqrt(np.sum((coordinates[v1] - coordinates[v2])**2))

                # Apply distance threshold if specified
                if distance_threshold is None or dist <= distance_threshold:
                    W[v1, v2] = 1
                    W[v2, v1] = 1
                    edges.add((min(v1, v2), max(v1, v2)))

    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums

    return W, list(edges)


def build_lattice_weights(nrows, ncols):
    """
    Build spatial weights for a regular lattice grid (queen contiguity).

    Parameters
    ----------
    nrows : int
        Number of rows in the grid
    ncols : int
        Number of columns in the grid

    Returns
    -------
    W : ndarray
        Row-standardized spatial weights matrix
    """
    n = nrows * ncols
    W = np.zeros((n, n))

    for i in range(n):
        row = i // ncols
        col = i % ncols

        # Queen contiguity: 8 neighbors (or fewer at edges)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    j = nr * ncols + nc
                    W[i, j] = 1

    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    return W


def compute_quadrat_density(coordinates, image_shape, grid_size=10):
    """
    Compute object density in a grid of quadrats.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape (n, 2) with (row, col) coordinates
    image_shape : tuple
        Shape of the original image (height, width)
    grid_size : int
        Number of grid cells along each dimension

    Returns
    -------
    density : ndarray
        2D array of object counts per quadrat
    quadrat_coords : ndarray
        Array of quadrat center coordinates, shape (grid_size^2, 2)
    """
    height, width = image_shape[:2]

    row_bins = np.linspace(0, height, grid_size + 1)
    col_bins = np.linspace(0, width, grid_size + 1)

    density, _, _ = np.histogram2d(
        coordinates[:, 0], coordinates[:, 1],
        bins=[row_bins, col_bins]
    )

    row_centers = (row_bins[:-1] + row_bins[1:]) / 2
    col_centers = (col_bins[:-1] + col_bins[1:]) / 2

    rr, cc = np.meshgrid(row_centers, col_centers, indexing='ij')
    quadrat_coords = np.column_stack([rr.ravel(), cc.ravel()])

    return density, quadrat_coords


def calculate_morans_i(values, W, permutations=999):
    """
    Calculate Global Moran's I statistic.

    Parameters
    ----------
    values : ndarray
        1D array of attribute values
    W : ndarray
        Spatial weights matrix (row-standardized)
    permutations : int
        Number of permutations for significance testing

    Returns
    -------
    dict
        Contains: morans_i, expected_i, z_score, p_value, p_value_sim
    """
    n = len(values)
    y = values - values.mean()

    # Calculate Moran's I
    numerator = np.sum(W * np.outer(y, y))
    denominator = np.sum(y ** 2)

    if denominator == 0:
        return {
            'morans_i': 0.0,
            'expected_i': -1 / (n - 1),
            'variance': 0.0,
            'z_score': 0.0,
            'p_value': 1.0,
            'p_value_sim': 1.0
        }

    I = (n / W.sum()) * (numerator / denominator)

    # Expected value under null hypothesis
    EI = -1 / (n - 1)

    # Variance under normality assumption
    S0 = W.sum()
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)

    b2 = (np.sum(y ** 4) / n) / ((np.sum(y ** 2) / n) ** 2)

    A = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)
    B = b2 * ((n**2 - n) * S1 - 2 * n * S2 + 6 * S0**2)
    C = (n - 1) * (n - 2) * (n - 3) * S0**2

    VI = (A - B) / C - EI**2

    # Z-score
    z = (I - EI) / np.sqrt(VI) if VI > 0 else 0

    # Two-tailed p-value from normal distribution
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Permutation test for significance
    p_value_sim = None
    if permutations > 0:
        I_perm = np.zeros(permutations)
        for p in range(permutations):
            y_perm = np.random.permutation(y)
            num_perm = np.sum(W * np.outer(y_perm, y_perm))
            I_perm[p] = (n / S0) * (num_perm / denominator)

        # Two-tailed test
        p_value_sim = (np.sum(np.abs(I_perm) >= np.abs(I)) + 1) / (permutations + 1)

    return {
        'morans_i': I,
        'expected_i': EI,
        'variance': VI,
        'z_score': z,
        'p_value': p_value,
        'p_value_sim': p_value_sim
    }


def calculate_local_morans(values, W, permutations=999):
    """
    Calculate Local Moran's I (LISA) statistics.

    Parameters
    ----------
    values : ndarray
        1D array of attribute values
    W : ndarray
        Spatial weights matrix (row-standardized)
    permutations : int
        Number of permutations for significance testing

    Returns
    -------
    dict
        Contains: local_i, p_values, quadrants, significant
    """
    n = len(values)
    y = values - values.mean()
    y_std = y / y.std() if y.std() > 0 else y

    # Spatial lag
    lag = W @ y_std

    # Local Moran's I
    local_i = y_std * lag

    # Quadrant classification
    # 1 = HH (high-high), 2 = LH (low-high), 3 = LL (low-low), 4 = HL (high-low)
    quadrants = np.zeros(n, dtype=int)
    quadrants[(y_std > 0) & (lag > 0)] = 1  # HH
    quadrants[(y_std < 0) & (lag > 0)] = 2  # LH
    quadrants[(y_std < 0) & (lag < 0)] = 3  # LL
    quadrants[(y_std > 0) & (lag < 0)] = 4  # HL

    # Permutation test for local significance
    p_values = np.ones(n)
    if permutations > 0:
        local_i_perm = np.zeros((permutations, n))
        for p in range(permutations):
            y_perm = np.random.permutation(y_std)
            lag_perm = W @ y_perm
            local_i_perm[p] = y_perm * lag_perm

        for i in range(n):
            # Conditional permutation approach
            p_values[i] = (np.sum(np.abs(local_i_perm[:, i]) >= np.abs(local_i[i])) + 1) / (permutations + 1)

    quadrant_names = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}

    return {
        'local_i': local_i,
        'p_values': p_values,
        'quadrants': quadrants,
        'quadrant_names': quadrant_names,
        'significant': p_values < 0.05
    }


def calculate_ripleys_k(coordinates, image_shape, radii=None, edge_correction=True,
                        max_points=5000):
    """
    Calculate Ripley's K function for point pattern analysis.

    K(r) measures the expected number of points within distance r of a typical point,
    normalized by intensity. K(r) > pi*r^2 indicates clustering, K(r) < pi*r^2 indicates dispersion.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape (n, 2) with (row, col) coordinates
    image_shape : tuple
        Shape of the study area (height, width)
    radii : ndarray, optional
        Array of distances at which to evaluate K. Default: 20 evenly spaced values
    edge_correction : bool
        Whether to apply Ripley's edge correction (default: True)
    max_points : int
        Maximum points to use (subsamples if n > max_points for performance)

    Returns
    -------
    dict
        Contains: radii, K, L, L_minus_r (L(r) - r for easier interpretation)
    """
    n = len(coordinates)
    height, width = image_shape[:2]
    area = height * width
    intensity = n / area

    # Subsample if too many points (O(n^2) complexity)
    if n > max_points:
        print(f"  Subsampling {max_points} of {n} points for Ripley's K...")
        idx = np.random.choice(n, max_points, replace=False)
        coords_sample = coordinates[idx]
        n_sample = max_points
    else:
        coords_sample = coordinates
        n_sample = n

    if radii is None:
        max_r = min(height, width) / 4
        radii = np.linspace(0, max_r, 20)

    # Compute all pairwise distances
    distances = cdist(coords_sample, coords_sample)

    K = np.zeros(len(radii))

    for i, r in enumerate(radii):
        if r == 0:
            K[i] = 0
            continue

        count = 0
        for j in range(n_sample):
            # Count points within distance r of point j (excluding self)
            neighbors = np.sum((distances[j] <= r) & (distances[j] > 0))

            if edge_correction:
                # Ripley's isotropic edge correction
                y, x = coords_sample[j]
                # Weight by proportion of circle inside study area
                weight = _edge_correction_weight(x, y, r, width, height)
                count += neighbors / weight if weight > 0 else neighbors
            else:
                count += neighbors

        K[i] = count / (n_sample * intensity)

    # L function: L(r) = sqrt(K(r)/pi) - easier to interpret
    L = np.sqrt(K / np.pi)

    # L(r) - r: positive = clustering, negative = dispersion, zero = random
    L_minus_r = L - radii

    return {
        'radii': radii,
        'K': K,
        'L': L,
        'L_minus_r': L_minus_r,
        'n_sampled': n_sample
    }


def _edge_correction_weight(x, y, r, width, height):
    """
    Calculate Ripley's isotropic edge correction weight.
    Returns the proportion of circle perimeter inside the study area.
    """
    # Simplified edge correction: proportion of area
    # More accurate would use arc length, but this is a good approximation
    inside = 1.0

    # Check distance to each edge
    if x < r:
        inside *= (0.5 + x / (2 * r))
    if x > width - r:
        inside *= (0.5 + (width - x) / (2 * r))
    if y < r:
        inside *= (0.5 + y / (2 * r))
    if y > height - r:
        inside *= (0.5 + (height - y) / (2 * r))

    return max(inside, 0.1)  # Avoid division by zero


def calculate_pair_correlation(coordinates, image_shape, radii=None, dr=None,
                                max_points=5000):
    """
    Calculate the pair correlation function g(r).

    g(r) is the derivative of Ripley's K, measuring the probability of finding
    a point at exactly distance r from another point, relative to random expectation.
    g(r) > 1 indicates clustering at distance r, g(r) < 1 indicates dispersion.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape (n, 2) with (row, col) coordinates
    image_shape : tuple
        Shape of the study area (height, width)
    radii : ndarray, optional
        Array of distances at which to evaluate g(r)
    dr : float, optional
        Width of annulus for estimation. Default: radii[1] - radii[0]
    max_points : int
        Maximum points to use (subsamples if n > max_points for performance)

    Returns
    -------
    dict
        Contains: radii, g (pair correlation values)
    """
    n = len(coordinates)
    height, width = image_shape[:2]
    area = height * width
    intensity = n / area

    # Subsample if too many points
    if n > max_points:
        print(f"  Subsampling {max_points} of {n} points for pair correlation...")
        idx = np.random.choice(n, max_points, replace=False)
        coords_sample = coordinates[idx]
        n_sample = max_points
    else:
        coords_sample = coordinates
        n_sample = n

    if radii is None:
        max_r = min(height, width) / 4
        radii = np.linspace(5, max_r, 20)  # Start at 5 to avoid r=0

    if dr is None:
        dr = radii[1] - radii[0] if len(radii) > 1 else radii[0] / 2

    # Compute all pairwise distances
    distances = cdist(coords_sample, coords_sample)

    g = np.zeros(len(radii))

    for i, r in enumerate(radii):
        # Count points in annulus [r - dr/2, r + dr/2]
        r_inner = max(0, r - dr / 2)
        r_outer = r + dr / 2

        count = 0
        for j in range(n_sample):
            neighbors = np.sum((distances[j] > r_inner) & (distances[j] <= r_outer))
            count += neighbors

        # Expected count under CSR (Complete Spatial Randomness)
        annulus_area = np.pi * (r_outer**2 - r_inner**2)
        expected = n_sample * (n_sample - 1) * annulus_area / area

        g[i] = count / expected if expected > 0 else 1.0

    return {
        'radii': radii,
        'g': g,
        'dr': dr,
        'n_sampled': n_sample
    }


def calculate_morans_i_centroids(coordinates, image_shape, distance_threshold=None,
                                  permutations=999, max_points=5000):
    """
    Calculate Moran's I directly on centroid positions using distance-based weights.

    Uses local density at each centroid as the attribute value, computed as
    the number of neighbors within the distance threshold.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape (n, 2) with (row, col) coordinates
    image_shape : tuple
        Shape of the study area (height, width)
    distance_threshold : float, optional
        Distance threshold for neighbors. Default: 10% of image diagonal
    permutations : int
        Number of permutations for significance testing
    max_points : int
        Maximum points to use (subsamples if n > max_points for performance)

    Returns
    -------
    dict
        Contains: morans_i, z_score, p_value, interpretation, local_density
    """
    n = len(coordinates)

    if n < 4:
        raise ValueError(f"Too few points ({n}) for Moran's I analysis")

    # Subsample if too many points
    if n > max_points:
        print(f"  Subsampling {max_points} of {n} points for centroid Moran's I...")
        idx = np.random.choice(n, max_points, replace=False)
        coords_sample = coordinates[idx]
        n_sample = max_points
    else:
        coords_sample = coordinates
        n_sample = n

    # Default distance threshold
    if distance_threshold is None:
        height, width = image_shape[:2]
        distance_threshold = np.sqrt(height**2 + width**2) * 0.1

    # Compute pairwise distances
    distances = cdist(coords_sample, coords_sample)

    # Compute local density for each point (number of neighbors within threshold)
    local_density = np.sum(distances <= distance_threshold, axis=1) - 1  # Exclude self

    # Build distance-based weights matrix
    W = build_spatial_weights_distance(coords_sample, distance_threshold)

    # Calculate Moran's I on local density values
    result = calculate_morans_i(local_density, W, permutations)

    # Add interpretation
    if result['p_value'] > 0.05:
        interpretation = 'random'
    elif result['morans_i'] > 0:
        interpretation = 'clustered'
    else:
        interpretation = 'dispersed'

    return {
        'morans_i': result['morans_i'],
        'expected_i': result['expected_i'],
        'z_score': result['z_score'],
        'p_value': result['p_value'],
        'p_value_sim': result['p_value_sim'],
        'interpretation': interpretation,
        'local_density': local_density,
        'distance_threshold': distance_threshold,
        'n_sampled': n_sample
    }


def calculate_morans_i_delaunay(coordinates, image_shape, distance_threshold=None,
                                 permutations=999, max_points=5000):
    """
    Calculate Moran's I on centroids using Delaunay triangulation-based spatial weights.

    Uses Delaunay triangulation to define neighbors. This creates a natural neighbor
    structure where points are connected if they share a Delaunay edge. An optional
    distance threshold can filter out long edges.

    Uses local density at each centroid as the attribute value.

    Parameters
    ----------
    coordinates : ndarray
        Array of shape (n, 2) with (row, col) coordinates
    image_shape : tuple
        Shape of the study area (height, width)
    distance_threshold : float, optional
        Maximum edge length to retain in the Delaunay graph. If None, all edges kept.
    permutations : int
        Number of permutations for significance testing
    max_points : int
        Maximum points to use (subsamples if n > max_points for performance)

    Returns
    -------
    dict
        Contains: morans_i, z_score, p_value, interpretation, local_density,
                  edges (Delaunay edges for visualization)
    """
    n = len(coordinates)

    if n < 4:
        raise ValueError(f"Too few points ({n}) for Moran's I analysis")

    # Subsample if too many points
    if n > max_points:
        print(f"  Subsampling {max_points} of {n} points for Delaunay Moran's I...")
        idx = np.random.choice(n, max_points, replace=False)
        coords_sample = coordinates[idx]
        n_sample = max_points
    else:
        coords_sample = coordinates
        n_sample = n

    # Build Delaunay-based weights
    W, edges = build_spatial_weights_delaunay(coords_sample, distance_threshold)

    # Compute local density from the Delaunay weights
    # (number of Delaunay neighbors, which is already encoded in W before standardization)
    n_neighbors = (W > 0).sum(axis=1)
    local_density = n_neighbors.astype(float)

    # Calculate Moran's I on local density values
    result = calculate_morans_i(local_density, W, permutations)

    # Add interpretation
    if result['p_value'] > 0.05:
        interpretation = 'random'
    elif result['morans_i'] > 0:
        interpretation = 'clustered'
    else:
        interpretation = 'dispersed'

    return {
        'morans_i': result['morans_i'],
        'expected_i': result['expected_i'],
        'z_score': result['z_score'],
        'p_value': result['p_value'],
        'p_value_sim': result['p_value_sim'],
        'interpretation': interpretation,
        'local_density': local_density,
        'distance_threshold': distance_threshold,
        'n_sampled': n_sample,
        'edges': edges,
        'n_edges': len(edges)
    }


def visualize_delaunay(coordinates, edges, image_shape, moran_result=None, save_path=None):
    """
    Visualize Delaunay triangulation network with optional Moran's I annotation.

    Parameters
    ----------
    coordinates : ndarray
        Object centroid coordinates
    edges : list
        List of (i, j) edge tuples from build_spatial_weights_delaunay
    image_shape : tuple
        Original image shape
    moran_result : dict, optional
        Results from calculate_morans_i_delaunay
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    height, width = image_shape[:2]

    # Draw edges
    for i, j in edges:
        ax.plot([coordinates[i, 1], coordinates[j, 1]],
                [coordinates[i, 0], coordinates[j, 0]],
                'b-', alpha=0.3, linewidth=0.5)

    # Draw points
    ax.scatter(coordinates[:, 1], coordinates[:, 0], s=10, c='red', alpha=0.7, zorder=5)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if moran_result:
        title = (f"Delaunay Triangulation Network\n"
                 f"n={len(coordinates)}, edges={moran_result['n_edges']}\n"
                 f"Moran's I = {moran_result['morans_i']:.4f}, "
                 f"p = {moran_result['p_value']:.4f} ({moran_result['interpretation']})")
    else:
        title = f"Delaunay Triangulation Network (n={len(coordinates)}, edges={len(edges)})"

    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Delaunay figure saved to: {save_path}")

    plt.show()
    return fig


def visualize_spatial_statistics(coordinates, image_shape, ripley_result=None,
                                  pcf_result=None, save_path=None):
    """
    Visualize Ripley's K/L and pair correlation function results.

    Parameters
    ----------
    coordinates : ndarray
        Object centroid coordinates
    image_shape : tuple
        Original image shape
    ripley_result : dict
        Results from calculate_ripleys_k
    pcf_result : dict
        Results from calculate_pair_correlation
    save_path : str, optional
        Path to save figure
    """
    n_plots = 1 + (ripley_result is not None) + (pcf_result is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    height, width = image_shape[:2]
    plot_idx = 0

    # Plot 1: Point pattern
    ax = axes[plot_idx]
    ax.scatter(coordinates[:, 1], coordinates[:, 0], s=5, alpha=0.5, c='blue')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.set_title(f'Point Pattern (n={len(coordinates)})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plot_idx += 1

    # Plot 2: Ripley's L function
    if ripley_result is not None:
        ax = axes[plot_idx]
        radii = ripley_result['radii']
        L_minus_r = ripley_result['L_minus_r']

        ax.plot(radii, L_minus_r, 'b-', linewidth=2, label="L(r) - r")
        ax.axhline(0, color='red', linestyle='--', linewidth=1, label='CSR expectation')
        ax.fill_between(radii, L_minus_r, 0, where=(L_minus_r > 0),
                        alpha=0.3, color='red', label='Clustering')
        ax.fill_between(radii, L_minus_r, 0, where=(L_minus_r < 0),
                        alpha=0.3, color='blue', label='Dispersion')
        ax.set_xlabel('Distance (r)')
        ax.set_ylabel("L(r) - r")
        ax.set_title("Ripley's L Function")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 3: Pair correlation function
    if pcf_result is not None:
        ax = axes[plot_idx]
        radii = pcf_result['radii']
        g = pcf_result['g']

        ax.plot(radii, g, 'b-', linewidth=2, label='g(r)')
        ax.axhline(1, color='red', linestyle='--', linewidth=1, label='CSR expectation')
        ax.fill_between(radii, g, 1, where=(g > 1),
                        alpha=0.3, color='red', label='Clustering')
        ax.fill_between(radii, g, 1, where=(g < 1),
                        alpha=0.3, color='blue', label='Dispersion')
        ax.set_xlabel('Distance (r)')
        ax.set_ylabel('g(r)')
        ax.set_title('Pair Correlation Function')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig


def interpret_morans_i(morans_i, p_value, alpha=0.05):
    """
    Interpret Moran's I result.
    """
    if p_value > alpha:
        return 'random'
    elif morans_i > 0:
        return 'clustered'
    else:
        return 'dispersed'


def visualize_results(coordinates, image_shape, density, quadrat_coords,
                      moran_result, W, local_moran=None, grid_size=10,
                      save_path=None):
    """
    Visualize spatial autocorrelation analysis results.
    """
    n_plots = 4 if local_moran is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    height, width = image_shape[:2]

    # Plot 1: Object positions
    ax = axes[0]
    ax.scatter(coordinates[:, 1], coordinates[:, 0], s=10, alpha=0.6, c='blue')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.set_title(f'Object Positions (n={len(coordinates)})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Plot 2: Quadrat density heatmap
    ax = axes[1]
    im = ax.imshow(density, cmap='YlOrRd', origin='upper',
                   extent=[0, width, height, 0])
    plt.colorbar(im, ax=ax, label='Count')
    ax.set_title('Quadrat Density')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Plot 3: Moran scatterplot
    ax = axes[2]
    values = density.ravel()

    if values.std() > 0:
        values_std = (values - values.mean()) / values.std()
    else:
        values_std = values - values.mean()

    lag = W @ values_std

    ax.scatter(values_std, lag, alpha=0.6)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    slope = moran_result['morans_i']
    x_range = np.array([values_std.min(), values_std.max()])
    ax.plot(x_range, slope * x_range, 'r-', linewidth=2,
            label=f"Moran's I = {slope:.3f}")
    ax.legend()
    ax.set_xlabel('Standardized Density')
    ax.set_ylabel('Spatial Lag')
    ax.set_title("Moran's I Scatterplot")

    # Plot 4: LISA cluster map
    if local_moran is not None:
        ax = axes[3]

        cluster_map = np.zeros_like(density)
        for i, (q, sig) in enumerate(zip(local_moran['quadrants'],
                                          local_moran['significant'])):
            row, col = i // grid_size, i % grid_size
            if sig:
                cluster_map[row, col] = q

        cmap = plt.cm.colors.ListedColormap(['white', 'red', 'lightblue',
                                              'blue', 'pink'])
        im = ax.imshow(cluster_map, cmap=cmap, origin='upper',
                       extent=[0, width, height, 0], vmin=0, vmax=4)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Not Significant'),
            Patch(facecolor='red', label='High-High (cluster)'),
            Patch(facecolor='lightblue', label='Low-High (outlier)'),
            Patch(facecolor='blue', label='Low-Low (cluster)'),
            Patch(facecolor='pink', label='High-Low (outlier)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax.set_title('LISA Cluster Map')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    interpretation = interpret_morans_i(moran_result['morans_i'],
                                         moran_result['p_value'])
    fig.suptitle(
        f"Spatial Autocorrelation Analysis: {interpretation.upper()}\n"
        f"Moran's I = {moran_result['morans_i']:.3f}, "
        f"z = {moran_result['z_score']:.2f}, "
        f"p = {moran_result['p_value']:.4f}",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig


def analyze_spatial_distribution(label_mask, weight_method='knn', k=8,
                                  distance_threshold=None, grid_size=10,
                                  permutations=999, visualize=True,
                                  save_path=None, min_area=0):
    """
    Complete spatial autocorrelation analysis pipeline.

    Analyzes whether objects in a labeled image are spatially clustered,
    dispersed, or randomly distributed using Moran's I index.

    Parameters
    ----------
    label_mask : ndarray
        2D array where each object has a unique integer ID (0 = background)
    weight_method : str
        'knn' for k-nearest neighbors, 'distance' for distance threshold,
        'delaunay' for Delaunay triangulation, or 'lattice' for grid-based (default)
    k : int
        Number of nearest neighbors (if weight_method='knn')
    distance_threshold : float
        Distance cutoff (if weight_method='distance' or 'delaunay')
    grid_size : int
        Number of quadrats along each image dimension
    permutations : int
        Number of permutations for significance testing
    visualize : bool
        Whether to display visualization
    save_path : str, optional
        Path to save visualization figure
    min_area : int
        Minimum object area in pixels to include (filters noise)

    Returns
    -------
    dict
        Results containing:
        - morans_i: Global Moran's I value (-1 to +1)
        - z_score: Standard deviations from expected under null
        - p_value: Statistical significance (two-tailed)
        - interpretation: 'clustered', 'dispersed', or 'random'
        - n_objects: Number of objects detected
        - centroids: Object centroid coordinates
        - density_grid: Quadrat density matrix
        - local_morans: Local Moran's I results (dict)
    """
    # Extract object centroids
    centroids, labels = extract_object_centroids(label_mask, min_area=min_area)
    n_objects = len(centroids)

    if n_objects < 4:
        raise ValueError(f"Too few objects ({n_objects}) for spatial analysis. "
                        "Need at least 4 objects.")

    if min_area > 0:
        print(f"Found {n_objects} objects in the image (min_area={min_area})")
    else:
        print(f"Found {n_objects} objects in the image")

    # Compute quadrat density
    image_shape = label_mask.shape
    density, quadrat_coords = compute_quadrat_density(
        centroids, image_shape, grid_size
    )

    # Build spatial weights based on method
    if weight_method == 'knn':
        W = build_spatial_weights_knn(quadrat_coords, k)
    elif weight_method == 'distance':
        W = build_spatial_weights_distance(quadrat_coords, distance_threshold)
    elif weight_method == 'delaunay':
        W, _ = build_spatial_weights_delaunay(quadrat_coords, distance_threshold)
    else:
        W = build_lattice_weights(grid_size, grid_size)

    # Flatten density for analysis
    density_flat = density.ravel()

    # Calculate Global Moran's I
    print("Computing Global Moran's I...")
    moran_result = calculate_morans_i(density_flat, W, permutations)

    # Calculate Local Moran's I
    print("Computing Local Moran's I (LISA)...")
    local_moran = calculate_local_morans(density_flat, W, permutations)

    # Interpret results
    interpretation = interpret_morans_i(
        moran_result['morans_i'],
        moran_result['p_value']
    )

    # Print summary
    print("\n" + "="*50)
    print("SPATIAL AUTOCORRELATION RESULTS")
    print("="*50)
    print(f"Number of objects: {n_objects}")
    print(f"Grid size: {grid_size} x {grid_size} quadrats")
    print(f"Moran's I: {moran_result['morans_i']:.4f}")
    print(f"Expected I: {moran_result['expected_i']:.4f}")
    print(f"Z-score: {moran_result['z_score']:.4f}")
    print(f"P-value: {moran_result['p_value']:.4f}")
    if moran_result['p_value_sim'] is not None:
        print(f"P-value (permutation): {moran_result['p_value_sim']:.4f}")
    print(f"\nInterpretation: {interpretation.upper()}")

    if interpretation == 'clustered':
        print("  -> Objects tend to be near other objects (positive autocorrelation)")
    elif interpretation == 'dispersed':
        print("  -> Objects tend to be far from other objects (negative autocorrelation)")
    else:
        print("  -> No significant spatial pattern detected")

    n_significant = np.sum(local_moran['significant'])
    print(f"\nLocal analysis: {n_significant} significant quadrats detected")
    print("="*50)

    if visualize:
        visualize_results(
            centroids, image_shape, density, quadrat_coords,
            moran_result, W, local_moran, grid_size, save_path
        )

    return {
        'morans_i': moran_result['morans_i'],
        'expected_i': moran_result['expected_i'],
        'z_score': moran_result['z_score'],
        'p_value': moran_result['p_value'],
        'p_value_sim': moran_result['p_value_sim'],
        'interpretation': interpretation,
        'n_objects': n_objects,
        'centroids': centroids,
        'density_grid': density,
        'local_morans': local_moran,
        'quadrat_coords': quadrat_coords
    }


def generate_random_control(n_objects, image_shape, object_size=5):
    """
    Generate a random control mask with the same number of objects.

    Parameters
    ----------
    n_objects : int
        Number of objects to place randomly
    image_shape : tuple
        Shape of the image (height, width)
    object_size : int
        Size of each object (default 5x5 pixels)

    Returns
    -------
    mask : ndarray
        Label mask with randomly placed objects
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.int32)
    margin = object_size // 2 + 1

    for label in range(1, n_objects + 1):
        x = np.random.randint(margin, height - margin)
        y = np.random.randint(margin, width - margin)
        r = object_size // 2
        mask[x-r:x+r+1, y-r:y+r+1] = label

    return mask


def analyze_with_random_control(label_mask, weight_method='lattice', k=8,
                                 distance_threshold=None, grid_size=10,
                                 permutations=999, save_path=None, min_area=0):
    """
    Analyze spatial distribution and compare with a random control.

    Runs Moran's I analysis on the sample and generates a random control
    with the same number of objects for comparison.

    Parameters
    ----------
    label_mask : ndarray
        2D array where each object has a unique integer ID (0 = background)
    weight_method : str
        'knn' for k-nearest neighbors, 'distance' for distance threshold,
        or 'lattice' for grid-based weights (default)
    k : int
        Number of nearest neighbors (if weight_method='knn')
    distance_threshold : float
        Distance cutoff (if weight_method='distance')
    grid_size : int
        Number of quadrats along each image dimension
    permutations : int
        Number of permutations for significance testing
    save_path : str, optional
        Base path for saving outputs (will create _sample.pdf, _random.pdf, and _results.csv)

    Returns
    -------
    dict
        Contains 'sample' and 'random_control' results
    """
    print("="*60)
    print("SAMPLE ANALYSIS")
    print("="*60)

    # Determine save paths
    sample_path = None
    random_path = None
    if save_path:
        base = save_path.replace('.pdf', '')
        sample_path = f"{base}_sample.pdf"
        random_path = f"{base}_random.pdf"

    # Analyze the sample
    sample_results = analyze_spatial_distribution(
        label_mask,
        weight_method=weight_method,
        k=k,
        distance_threshold=distance_threshold,
        grid_size=grid_size,
        permutations=permutations,
        save_path=sample_path,
        min_area=min_area
    )

    # Generate random control with same number of objects
    n_objects = sample_results['n_objects']
    image_shape = label_mask.shape

    print("\n")
    print("="*60)
    print(f"RANDOM CONTROL (n={n_objects} objects)")
    print("="*60)

    random_mask = generate_random_control(n_objects, image_shape)
    random_results = analyze_spatial_distribution(
        random_mask,
        weight_method=weight_method,
        k=k,
        distance_threshold=distance_threshold,
        grid_size=grid_size,
        permutations=permutations,
        save_path=random_path,
        min_area=0  # Random control doesn't need filtering
    )

    # Print comparison summary
    print("\n")
    print("="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'':20} {'Sample':>12} {'Random Ctrl':>12}")
    print("-"*60)
    print(f"{'N objects':20} {sample_results['n_objects']:>12} {random_results['n_objects']:>12}")
    print(f"{'Moran I':20} {sample_results['morans_i']:>12.4f} {random_results['morans_i']:>12.4f}")
    print(f"{'Z-score':20} {sample_results['z_score']:>12.4f} {random_results['z_score']:>12.4f}")
    print(f"{'P-value':20} {sample_results['p_value']:>12.4f} {random_results['p_value']:>12.4f}")
    print(f"{'Interpretation':20} {sample_results['interpretation']:>12} {random_results['interpretation']:>12}")
    print("="*60)

    # Save results to CSV
    if save_path:
        csv_path = save_path.replace('.pdf', '') + '_results.csv'
        with open(csv_path, 'w') as f:
            f.write("Metric,Sample,Random_Control\n")
            f.write(f"N_objects,{sample_results['n_objects']},{random_results['n_objects']}\n")
            f.write(f"Morans_I,{sample_results['morans_i']:.6f},{random_results['morans_i']:.6f}\n")
            f.write(f"Expected_I,{sample_results['expected_i']:.6f},{random_results['expected_i']:.6f}\n")
            f.write(f"Z_score,{sample_results['z_score']:.6f},{random_results['z_score']:.6f}\n")
            f.write(f"P_value,{sample_results['p_value']:.6f},{random_results['p_value']:.6f}\n")
            f.write(f"P_value_permutation,{sample_results['p_value_sim']:.6f},{random_results['p_value_sim']:.6f}\n")
            f.write(f"Interpretation,{sample_results['interpretation']},{random_results['interpretation']}\n")
        print(f"\nResults saved to: {csv_path}")

    return {
        'sample': sample_results,
        'random_control': random_results
    }


def analyze_comprehensive(label_mask, grid_size=10, permutations=999,
                          distance_threshold=None, save_path=None, min_area=0):
    """
    Comprehensive spatial analysis including all statistics.

    Computes:
    - Moran's I (quadrat-based)
    - Moran's I (centroid-based with distance weights)
    - Moran's I (centroid-based with Delaunay triangulation weights)
    - Ripley's K and L functions
    - Pair correlation function g(r)
    - Random control comparison

    Parameters
    ----------
    label_mask : ndarray
        2D array where each object has a unique integer ID (0 = background)
    grid_size : int
        Number of quadrats along each dimension
    permutations : int
        Number of permutations for significance testing
    distance_threshold : float, optional
        Distance threshold for centroid-based analysis
    save_path : str, optional
        Base path for saving outputs
    min_area : int
        Minimum object area in pixels to include (filters noise)

    Returns
    -------
    dict
        Comprehensive results including all statistics
    """
    # Extract centroids
    centroids, labels = extract_object_centroids(label_mask, min_area=min_area)
    n_objects = len(centroids)
    image_shape = label_mask.shape

    if n_objects < 4:
        raise ValueError(f"Too few objects ({n_objects}) for analysis")

    print("="*60)
    print("COMPREHENSIVE SPATIAL ANALYSIS")
    if min_area > 0:
        print(f"(min_area filter: {min_area} pixels)")
    print("="*60)
    print(f"Number of objects: {n_objects}")
    print(f"Image shape: {image_shape}")

    # Determine save paths
    base = save_path.replace('.pdf', '') if save_path else None

    # 1. Moran's I (quadrat-based)
    print("\n" + "-"*40)
    print("1. MORAN'S I (Quadrat-based)")
    print("-"*40)
    density, quadrat_coords = compute_quadrat_density(centroids, image_shape, grid_size)
    W_lattice = build_lattice_weights(grid_size, grid_size)
    moran_quadrat = calculate_morans_i(density.ravel(), W_lattice, permutations)
    moran_quadrat_interp = interpret_morans_i(moran_quadrat['morans_i'], moran_quadrat['p_value'])
    print(f"Moran's I: {moran_quadrat['morans_i']:.4f}")
    print(f"Z-score: {moran_quadrat['z_score']:.4f}")
    print(f"P-value: {moran_quadrat['p_value']:.4f}")
    print(f"Interpretation: {moran_quadrat_interp.upper()}")

    # 2. Moran's I (centroid-based, distance weights)
    print("\n" + "-"*40)
    print("2. MORAN'S I (Centroid-based, distance weights)")
    print("-"*40)
    moran_centroid = calculate_morans_i_centroids(centroids, image_shape,
                                                   distance_threshold, permutations)
    print(f"Distance threshold: {moran_centroid['distance_threshold']:.1f}")
    print(f"Moran's I: {moran_centroid['morans_i']:.4f}")
    print(f"Z-score: {moran_centroid['z_score']:.4f}")
    print(f"P-value: {moran_centroid['p_value']:.4f}")
    print(f"Interpretation: {moran_centroid['interpretation'].upper()}")

    # 3. Moran's I (Delaunay triangulation)
    print("\n" + "-"*40)
    print("3. MORAN'S I (Delaunay triangulation)")
    print("-"*40)
    moran_delaunay = calculate_morans_i_delaunay(centroids, image_shape,
                                                  distance_threshold, permutations)
    print(f"Delaunay edges: {moran_delaunay['n_edges']}")
    if distance_threshold:
        print(f"Distance threshold: {distance_threshold:.1f}")
    print(f"Moran's I: {moran_delaunay['morans_i']:.4f}")
    print(f"Z-score: {moran_delaunay['z_score']:.4f}")
    print(f"P-value: {moran_delaunay['p_value']:.4f}")
    print(f"Interpretation: {moran_delaunay['interpretation'].upper()}")

    # 4. Ripley's K/L
    print("\n" + "-"*40)
    print("4. RIPLEY'S K/L FUNCTION")
    print("-"*40)
    ripley = calculate_ripleys_k(centroids, image_shape)
    max_L = np.max(ripley['L_minus_r'])
    min_L = np.min(ripley['L_minus_r'])
    r_max_cluster = ripley['radii'][np.argmax(ripley['L_minus_r'])]
    print(f"Max L(r)-r: {max_L:.2f} at r={r_max_cluster:.1f} (clustering)")
    print(f"Min L(r)-r: {min_L:.2f} (dispersion)")
    if max_L > 0:
        print("Interpretation: CLUSTERING detected at short distances")
    elif min_L < 0:
        print("Interpretation: DISPERSION detected")
    else:
        print("Interpretation: Consistent with RANDOM distribution")

    # 5. Pair correlation
    print("\n" + "-"*40)
    print("5. PAIR CORRELATION FUNCTION g(r)")
    print("-"*40)
    pcf = calculate_pair_correlation(centroids, image_shape)
    max_g = np.max(pcf['g'])
    r_max_g = pcf['radii'][np.argmax(pcf['g'])]
    print(f"Max g(r): {max_g:.2f} at r={r_max_g:.1f}")
    print(f"g(r) > 1 indicates clustering, g(r) < 1 indicates dispersion")

    # 6. Random control comparison
    print("\n" + "-"*40)
    print("6. RANDOM CONTROL COMPARISON")
    print("-"*40)
    random_mask = generate_random_control(n_objects, image_shape)
    random_centroids, _ = extract_object_centroids(random_mask)
    random_density, _ = compute_quadrat_density(random_centroids, image_shape, grid_size)
    moran_random = calculate_morans_i(random_density.ravel(), W_lattice, permutations)
    ripley_random = calculate_ripleys_k(random_centroids, image_shape)
    print(f"Random Moran's I: {moran_random['morans_i']:.4f} (p={moran_random['p_value']:.4f})")
    print(f"Random max L(r)-r: {np.max(ripley_random['L_minus_r']):.2f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Statistic':<30} {'Sample':>12} {'Random':>12}")
    print("-"*60)
    print(f"{'Morans I (quadrat)':30} {moran_quadrat['morans_i']:>12.4f} {moran_random['morans_i']:>12.4f}")
    print(f"{'Morans I (centroid)':30} {moran_centroid['morans_i']:>12.4f} {'N/A':>12}")
    print(f"{'Morans I (Delaunay)':30} {moran_delaunay['morans_i']:>12.4f} {'N/A':>12}")
    print(f"{'Max L(r)-r':30} {max_L:>12.2f} {np.max(ripley_random['L_minus_r']):>12.2f}")
    print(f"{'Max g(r)':30} {max_g:>12.2f} {'~1.0':>12}")
    print("="*60)

    # Visualize
    if save_path:
        # Main Moran's I visualization
        local_moran = calculate_local_morans(density.ravel(), W_lattice, permutations)
        visualize_results(centroids, image_shape, density, quadrat_coords,
                         moran_quadrat, W_lattice, local_moran, grid_size,
                         f"{base}_morans.pdf")

        # Ripley's K/L and PCF visualization
        visualize_spatial_statistics(centroids, image_shape, ripley, pcf,
                                     f"{base}_spatial_stats.pdf")

        # Delaunay triangulation visualization
        visualize_delaunay(centroids, moran_delaunay['edges'], image_shape,
                          moran_delaunay, f"{base}_delaunay.pdf")

        # Save CSV with all results
        csv_path = f"{base}_results.csv"
        with open(csv_path, 'w') as f:
            f.write("Statistic,Value,P_value,Interpretation\n")
            f.write(f"N_objects,{n_objects},,\n")
            f.write(f"Morans_I_quadrat,{moran_quadrat['morans_i']:.6f},{moran_quadrat['p_value']:.6f},{moran_quadrat_interp}\n")
            f.write(f"Morans_I_centroid,{moran_centroid['morans_i']:.6f},{moran_centroid['p_value']:.6f},{moran_centroid['interpretation']}\n")
            f.write(f"Morans_I_delaunay,{moran_delaunay['morans_i']:.6f},{moran_delaunay['p_value']:.6f},{moran_delaunay['interpretation']}\n")
            f.write(f"Delaunay_edges,{moran_delaunay['n_edges']},,\n")
            f.write(f"Ripleys_L_max,{max_L:.6f},,{'clustered' if max_L > 0 else 'random'}\n")
            f.write(f"Ripleys_R_at_max,{r_max_cluster:.2f},,\n")
            f.write(f"Max_g_r,{max_g:.6f},,{'clustered' if max_g > 1 else 'random'}\n")
            f.write(f"Random_Morans_I,{moran_random['morans_i']:.6f},{moran_random['p_value']:.6f},random\n")
        print(f"\nResults saved to: {csv_path}")

    return {
        'n_objects': n_objects,
        'centroids': centroids,
        'moran_quadrat': moran_quadrat,
        'moran_quadrat_interpretation': moran_quadrat_interp,
        'moran_centroid': moran_centroid,
        'moran_delaunay': moran_delaunay,
        'ripley': ripley,
        'pair_correlation': pcf,
        'random_moran': moran_random,
        'random_ripley': ripley_random
    }


if __name__ == '__main__':
    print("Running demo with synthetic clustered data...")

    np.random.seed(42)
    mask = np.zeros((512, 512), dtype=np.int32)

    cluster_centers = [(128, 128), (384, 384), (128, 384)]
    label = 1

    for cx, cy in cluster_centers:
        n_in_cluster = np.random.randint(20, 40)
        for _ in range(n_in_cluster):
            x = int(np.clip(cx + np.random.randn() * 30, 5, 507))
            y = int(np.clip(cy + np.random.randn() * 30, 5, 507))
            mask[x-2:x+3, y-2:y+3] = label
            label += 1

    results = analyze_with_random_control(mask, grid_size=8, permutations=99)
