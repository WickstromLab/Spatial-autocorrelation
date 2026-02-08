# Spatial Autocorrelation Analysis with Moran's I Index

## Overview
Python tool to analyze whether objects in microscopy images are spatially clustered, dispersed, or randomly distributed using Moran's I index.

**Input types supported:**
- **Binary images**: Binarized fluorescence microscopy images (0/1 or 0/255)
- **Label masks**: Pre-segmented images where each object has a unique integer ID

**Output**: Moran's I statistic, z-score, p-value, visualization (PDF), and random control comparison

---

## Installation

```bash
pip install numpy scipy scikit-image matplotlib
```

---

## Quick Start

### Analyzing a binarized fluorescence image

```python
from skimage import io
from skimage.measure import label
from spatial_autocorrelation import analyze_with_random_control

# Load binarized fluorescence image
binary_img = io.imread('binarized_fluorescence.tif')

# Convert binary to label mask (each connected component gets unique ID)
label_mask = label(binary_img > 0)

# Run analysis with automatic random control comparison
results = analyze_with_random_control(
    label_mask,
    grid_size=10,
    permutations=999,
    save_path='my_analysis.pdf'  # Creates _sample.pdf and _random.pdf
)

print(f"Sample Moran's I: {results['sample']['morans_i']:.3f}")
print(f"Random Control Moran's I: {results['random_control']['morans_i']:.3f}")
```

### Analyzing a pre-segmented label mask

```python
from skimage import io
from spatial_autocorrelation import analyze_spatial_distribution

# Load pre-segmented label mask (each object already has unique ID)
label_mask = io.imread('segmented_cells.tif')

# Run analysis
results = analyze_spatial_distribution(
    label_mask,
    grid_size=10,
    permutations=999,
    save_path='analysis.pdf'
)

print(f"Moran's I: {results['morans_i']:.3f}")
print(f"Pattern: {results['interpretation']}")
```

---

## Core Functions

### `analyze_with_random_control(label_mask, ...)` (Recommended)
Complete analysis with automatic random control comparison.

**Parameters:**
- `label_mask`: 2D numpy array (labeled image)
- `grid_size`: Number of quadrats along each dimension (default: 10)
- `permutations`: Number of permutations for significance testing (default: 999)
- `save_path`: Base path for saving figures (creates `*_sample.pdf` and `*_random.pdf`)

**Returns:** Dictionary with `sample` and `random_control` results

### `analyze_spatial_distribution(label_mask, ...)`
Single sample analysis without random control.

**Parameters:**
- `label_mask`: 2D numpy array (labeled image from segmentation or binary conversion)
- `weight_method`: 'knn' or 'distance'
- `k`: Number of nearest neighbors (default: 8)
- `distance_threshold`: Distance cutoff for distance-based weights
- `grid_size`: Number of quadrats along each dimension (default: 10)
- `permutations`: Number of permutations for significance testing (default: 999)
- `visualize`: Whether to display plots (default: True)
- `save_path`: Path to save figure (PDF recommended)

**Returns:** Dictionary with:
- `morans_i`: Global Moran's I value (-1 to +1)
- `z_score`: Standard deviations from expected
- `p_value`: Statistical significance
- `interpretation`: 'clustered', 'dispersed', or 'random'
- `n_objects`: Number of objects detected
- `local_morans`: Local I values per quadrat

### `generate_random_control(n_objects, image_shape)`
Generate a random control mask with specified number of objects.

---

## Interpretation Guide

| Moran's I | Meaning |
|-----------|---------|
| ≈ +1 | Strong clustering (objects grouped together) |
| ≈ 0 | Random distribution |
| ≈ -1 | Strong dispersion (objects evenly spaced apart) |

**Statistical significance:** p-value < 0.05 indicates a significant spatial pattern.

**Random control comparison:** Comparing your sample to a random control with the same number of objects confirms that any detected pattern is not simply due to object density.

---

## Visualizations

The analysis produces four plots (saved as PDF):
1. **Object Positions**: Scatter plot of object centroids
2. **Quadrat Density**: Heatmap of object counts per grid cell
3. **Moran Scatterplot**: Spatial lag vs. standardized value (slope = Moran's I)
4. **LISA Cluster Map**: Local clusters and outliers (High-High, Low-Low, etc.)

---

## Typical Workflow for Fluorescence Microscopy

1. **Acquire** fluorescence microscopy image
2. **Binarize** the image (threshold to separate objects from background)
3. **Label** connected components: `label_mask = skimage.measure.label(binary > 0)`
4. **Analyze** with `analyze_with_random_control(label_mask, save_path='output.pdf')`
5. **Interpret** results by comparing sample Moran's I to random control

---

## Files

| File | Purpose |
|------|---------|
| `spatial_autocorrelation.py` | Main module with all analysis functions |
| `example_usage.py` | Demo script with synthetic data examples |
| `README.md` | This documentation |
