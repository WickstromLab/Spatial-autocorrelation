"""
Example usage of spatial autocorrelation analysis with Moran's I.

This script demonstrates how to analyze the spatial distribution of
objects in microscopy images using the spatial_autocorrelation module.
"""

import numpy as np
from skimage import io
from spatial_autocorrelation import analyze_spatial_distribution


def create_clustered_data(size=512, n_clusters=4, objects_per_cluster=30):
    """Create a synthetic label mask with clustered objects."""
    mask = np.zeros((size, size), dtype=np.int32)
    label = 1

    # Random cluster centers
    np.random.seed(42)
    centers = np.random.randint(50, size - 50, size=(n_clusters, 2))

    for cx, cy in centers:
        for _ in range(objects_per_cluster):
            x = int(np.clip(cx + np.random.randn() * 25, 3, size - 4))
            y = int(np.clip(cy + np.random.randn() * 25, 3, size - 4))
            # Create small circular object
            mask[x-2:x+3, y-2:y+3] = label
            label += 1

    return mask


def create_random_data(size=512, n_objects=100):
    """Create a synthetic label mask with randomly distributed objects."""
    mask = np.zeros((size, size), dtype=np.int32)

    np.random.seed(42)
    for label in range(1, n_objects + 1):
        x = np.random.randint(3, size - 4)
        y = np.random.randint(3, size - 4)
        mask[x-2:x+3, y-2:y+3] = label

    return mask


def create_dispersed_data(size=512, grid_spacing=50):
    """Create a synthetic label mask with regularly dispersed objects."""
    mask = np.zeros((size, size), dtype=np.int32)
    label = 1

    # Place objects on a regular grid with small jitter
    np.random.seed(42)
    for x in range(grid_spacing // 2, size, grid_spacing):
        for y in range(grid_spacing // 2, size, grid_spacing):
            jx = x + np.random.randint(-5, 6)
            jy = y + np.random.randint(-5, 6)
            jx = np.clip(jx, 3, size - 4)
            jy = np.clip(jy, 3, size - 4)
            mask[jx-2:jx+3, jy-2:jy+3] = label
            label += 1

    return mask


def example_with_real_image():
    """
    Example: Load and analyze a real segmented image.

    Replace 'your_segmented_image.tif' with your actual file path.
    The image should be a label mask where:
    - Background pixels = 0
    - Each object has a unique positive integer ID
    """
    # Load your segmented label mask
    # label_mask = io.imread('your_segmented_image.tif')

    # Run analysis
    # results = analyze_spatial_distribution(
    #     label_mask,
    #     weight_method='knn',    # or 'distance'
    #     k=8,                     # neighbors for KNN
    #     grid_size=10,            # quadrat grid resolution
    #     permutations=999,        # for significance testing
    #     visualize=True,
    #     save_path='moran_analysis.png'
    # )

    # Access results
    # print(f"Moran's I: {results['morans_i']:.3f}")
    # print(f"Pattern: {results['interpretation']}")
    pass


def main():
    """Run demo analyses with synthetic data."""
    print("=" * 60)
    print("SPATIAL AUTOCORRELATION ANALYSIS DEMO")
    print("=" * 60)

    # Example 1: Clustered distribution
    print("\n\n>>> EXAMPLE 1: CLUSTERED DISTRIBUTION")
    print("-" * 40)
    clustered_mask = create_clustered_data()
    results_clustered = analyze_spatial_distribution(
        clustered_mask,
        grid_size=8,
        permutations=99,
        save_path='clustered_analysis.pdf'
    )

    # Get the number of objects from the clustered sample
    n_objects_sample = results_clustered['n_objects']
    print(f"\n(Using n={n_objects_sample} objects for random comparison)")

    # Example 2: Random distribution - use same number of objects as clustered
    print("\n\n>>> EXAMPLE 2: RANDOM DISTRIBUTION")
    print("-" * 40)
    random_mask = create_random_data(n_objects=n_objects_sample)
    results_random = analyze_spatial_distribution(
        random_mask,
        grid_size=8,
        permutations=99,
        save_path='random_analysis.pdf'
    )

    # Example 3: Dispersed distribution
    print("\n\n>>> EXAMPLE 3: DISPERSED DISTRIBUTION")
    print("-" * 40)
    # Calculate grid spacing to get approximately the same number of objects
    import math
    grid_spacing = int(512 / math.sqrt(n_objects_sample))
    dispersed_mask = create_dispersed_data(grid_spacing=grid_spacing)
    results_dispersed = analyze_spatial_distribution(
        dispersed_mask,
        grid_size=8,
        permutations=99,
        save_path='dispersed_analysis.pdf'
    )

    # Summary comparison
    print("\n\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Pattern':<15} {'Moran I':>10} {'Z-score':>10} {'P-value':>10} {'Result':>12}")
    print("-" * 60)

    for name, res in [('Clustered', results_clustered),
                      ('Random', results_random),
                      ('Dispersed', results_dispersed)]:
        print(f"{name:<15} {res['morans_i']:>10.3f} {res['z_score']:>10.2f} "
              f"{res['p_value']:>10.4f} {res['interpretation']:>12}")


if __name__ == '__main__':
    main()
