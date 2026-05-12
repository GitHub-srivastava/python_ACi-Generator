# Synthetic A–Ci Curve Data Generator for GNN-Based Limitation Classification

This repository contains code for generating synthetic CO₂ response curves (A–Ci curves) for graph neural network (GNN)-based node-level classification of photosynthetic limitation states.

The workflow was originally developed in MATLAB and then converted to Python/Jupyter Notebook format. The generated data can be used to train models that classify each sampled point on an A–Ci curve into its biochemical limitation state.

## Overview

The code simulates C3 photosynthesis over a range of intercellular CO₂ concentration (`Ci`) values using randomly sampled physiological parameters. From each simulated curve, a subset of points is selected using a uniform selection strategy. Each selected point is assigned a limitation-state label based on the dominant limiting photosynthetic rate.

The generated dataset contains:

- sampled `Ci` values,
- corresponding net photosynthesis values (`A_net`),
- node-level limitation labels,
- physiological parameters used to generate each curve.

This makes the dataset suitable for graph-based learning, where each A–Ci curve can be represented as a graph and each measurement point can be treated as a node.

## Repository Contents

| File | Description |
|---|---|
| `callInitialize_ACi.m` | MATLAB function for initializing leaf-state and mass-flux variables before photosynthesis simulation. |
| `callC3_ACi.m` | MATLAB implementation of the C3 photosynthesis calculation for a given input condition. |
| `callUniformSelect.m` | MATLAB function for selecting a subset of points from a generated A–Ci curve and assigning limitation labels. |
| `dataGeneratorUniform_forGNN.m` | Main MATLAB script for generating synthetic A–Ci curve datasets. |
| `ACi_GNN_MATLAB_to_Python.ipynb` | Python/Jupyter Notebook version of the MATLAB workflow. |
| `ACi_GNN_MATLAB_to_Python_allow_2ids.ipynb` | Updated notebook version that allows selected curves to contain two limitation classes instead of always requiring all three. |

Depending on the current repository version, file names may differ slightly.

## Main Functions

### `callInitialize_ACi`

Initializes the variables required for C3 photosynthesis simulation.

It creates the required data structures for:

- leaf mass fluxes,
- leaf-state variables,
- photosynthetic parameters at the given environmental condition.

In the Python notebook, this function is implemented as:

```python
callInitialize_ACi(Input, Photosynthesis)
```

### `callC3_ACi`

Computes the C3 photosynthetic rates for each input condition.

The function evaluates the major biochemical limitations:

- Rubisco-limited assimilation,
- RuBP-regeneration-limited assimilation,
- TPU-limited assimilation.

The net photosynthetic rate is then calculated after accounting for respiration.

In the Python notebook, this function is implemented as:

```python
callC3_ACi(Constants, Photosynthesis, Input_row, LeafState_row, LeafMassFlux_row)
```

### `callUniformSelect`

Selects a subset of points from a full simulated A–Ci curve.

The selected points are approximately distributed across the full `Ci` range. Each point is assigned a limitation label based on which photosynthetic rate controls the gross assimilation rate.

Label convention:

| Label | Limitation state |
|---|---|
| `1` | Rubisco-limited |
| `2` | RuBP-regeneration-limited |
| `3` | TPU-limited |

The updated version allows selected curves to contain at least two limitation classes. This means a curve does not need to contain all three classes to be accepted.

```python
callUniformSelect(
    LeafMassFlux,
    LeafState,
    Input,
    Photosynthesis,
    requiredpoints,
    Data,
    t,
    rng=None,
    min_curve_ids=2,
    min_selected_ids=2,
)
```

### `dataGeneratorUniform_forGNN`

Main dataset-generation function.

It repeatedly generates synthetic A–Ci curves by randomly sampling physiological parameters and environmental conditions. Each valid curve is then stored in the output dataset.

```python
Data = dataGeneratorUniform_forGNN(
    Number_of_curves=6000,
    output_dir="GeneratedData/GNN_Data",
    seed=42,
)
```

### `dataGeneratorUniform_forGNN_parallel`

Parallel version of the data generator.

This version generates full A–Ci curves independently across CPU cores. It is preferred for large datasets because each worker handles one complete valid curve.

```python
Data = dataGeneratorUniform_forGNN_parallel(
    Number_of_curves=6000,
    output_dir="GeneratedData/GNN_Data",
    seed=42,
    n_jobs=-1,
)
```

Here, `n_jobs=-1` uses all available CPU cores.

To use a fixed number of cores:

```python
Data = dataGeneratorUniform_forGNN_parallel(
    Number_of_curves=6000,
    output_dir="GeneratedData/GNN_Data",
    seed=42,
    n_jobs=4,
)
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy pandas scipy joblib jupyter
```

If using Google Colab, most packages are already available. Install missing packages using:

```python
!pip install joblib
```

## Running the Notebook

Open the notebook:

```bash
jupyter notebook ACi_GNN_MATLAB_to_Python_allow_2ids.ipynb
```

Then run the cells in order.

For a small test run:

```python
Data = dataGeneratorUniform_forGNN_parallel(
    Number_of_curves=10,
    output_dir="GeneratedData/GNN_Data",
    seed=42,
    n_jobs=2,
)
```

For full dataset generation:

```python
Data = dataGeneratorUniform_forGNN_parallel(
    Number_of_curves=6000,
    output_dir="GeneratedData/GNN_Data",
    seed=42,
    n_jobs=-1,
)
```

## Output Files

When `output_dir` is provided, the generator writes the following files:

| Output file | Description |
|---|---|
| `n3data_Ci.txt` | Selected intercellular CO₂ concentration values for each curve. |
| `n3data_Anet.txt` | Corresponding net photosynthesis values. |
| `n3data_ID.txt` | Limitation-state label for each selected point. |
| `n3data_Params.txt` | Photosynthetic and environmental parameters used to generate each curve. |

Each row corresponds to one generated A–Ci curve.

## Reproducibility

The Python version supports reproducible data generation using a random seed.

```python
seed=42
```

For parallel generation, independent child seeds are created using:

```python
np.random.SeedSequence(seed)
```

This ensures that each worker receives an independent random stream while keeping the overall run reproducible.

## Notes on Parallel Execution

The original MATLAB code uses `parfor` for parallel computation. In the Python version, parallelization is done at the curve level using `joblib`.

This is more efficient than parallelizing individual `Ci` points because each synthetic A–Ci curve is independent.

```python
from joblib import Parallel, delayed
```

The recommended parallel function is:

```python
dataGeneratorUniform_forGNN_parallel(...)
```

## Data Structure

The returned `Data` object is a Python dictionary:

```python
Data = {
    "ci": [...],
    "aNet": [...],
    "ID": [...],
    "paramArray": [...],
}
```

where:

- `Data["ci"]` contains selected `Ci` values,
- `Data["aNet"]` contains selected net photosynthesis values,
- `Data["ID"]` contains node-level limitation labels,
- `Data["paramArray"]` contains the parameters used to generate each curve.

Because the number of selected points can vary between curves, the stored arrays are ragged lists rather than a single rectangular matrix.

## Example Use Case for GNNs

Each generated A–Ci curve can be converted into a graph:

- nodes: selected measurement points,
- node features: `Ci`, `A_net`, and optionally auxiliary curve-derived signals,
- node labels: limitation-state IDs,
- edges: k-nearest-neighbor or auxiliary-signal-guided connections.

This enables node-level classification of photosynthetic limitation states using GNN models such as GCN, GAT, or Graph U-Net.

## Citation / Acknowledgement

If you use or extend this code, please cite or acknowledge the original project/code source appropriately.
