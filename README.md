# Synthetic A–Ci Curve Data Generator for GNN-Based Limitation Classification

This repository contains Python code for generating synthetic CO₂ response curves, commonly known as A–Ci curves, for graph neural network (GNN)-based node-level classification of photosynthetic limitation states.

The generated data can be used to train machine learning or graph-based models that identify the dominant biochemical limitation at each sampled point on an A–Ci curve.

## Overview

The code simulates C3 photosynthesis across a range of intercellular CO₂ concentration (`Ci`) values. For each generated curve, physiological parameters are randomly sampled from predefined ranges. The model then calculates photosynthetic rates and selects a subset of points from the full curve.

Each selected point is assigned a limitation-state label based on the dominant photosynthetic limitation.

The generated dataset includes:

- selected `Ci` values,
- corresponding net photosynthesis values (`A_net`),
- node-level limitation labels,
- physiological and environmental parameters used to generate each curve.

This dataset is designed for graph-based learning, where each A–Ci curve can be represented as a graph and each sampled measurement point can be treated as a node.

## Repository Contents

| File | Description |
|---|---|
| `ACi_GNN_MATLAB_to_Python.ipynb` | Main Python notebook for synthetic A–Ci curve generation. |
| `ACi_GNN_MATLAB_to_Python_allow_2ids.ipynb` | Updated Python notebook that allows selected curves to contain at least two limitation classes. |
| `README.md` | Project documentation. |

If file names are changed in the repository, use the notebook containing the latest version of the data generator and parallel execution functions.

## Main Workflow

The general workflow is:

1. Randomly sample physiological parameters.
2. Generate a full A–Ci curve over a predefined `Ci` range.
3. Calculate C3 photosynthetic rates at each `Ci` point.
4. Assign a biochemical limitation label to each point.
5. Select a subset of points from the full curve.
6. Store the selected `Ci`, `A_net`, class labels, and parameters.
7. Optionally generate curves in parallel across CPU cores.

## Limitation-State Labels

Each selected point is assigned one of the following labels:

| Label | Limitation state |
|---|---|
| `1` | Rubisco-limited |
| `2` | RuBP-regeneration-limited |
| `3` | TPU-limited |

The updated selection logic accepts curves that contain at least two limitation classes after point selection. This means the selected curve does not always need to contain all three classes.

For example, the following label sequences are acceptable:

```python
[1, 1, 1, 2, 2, 2, 2, 2]
```

```python
[1, 1, 2, 2, 2, 3, 3, 3]
```

Curves containing only one limitation class are rejected by default.

## Main Functions

### `callInitialize_ACi`

Initializes the required data structures for leaf-state variables and photosynthetic mass fluxes.

```python
LeafMassFlux, LeafState = callInitialize_ACi(Input, Photosynthesis)
```

### `callC3_ACi`

Calculates the C3 photosynthesis rates for a given environmental and physiological condition.

This function evaluates:

- Rubisco-limited assimilation,
- RuBP-regeneration-limited assimilation,
- TPU-limited assimilation,
- net photosynthesis after accounting for respiration.

```python
updated_mass_flux, updated_state = callC3_ACi(
    Constants,
    Photosynthesis,
    Input_row,
    LeafState_row,
    LeafMassFlux_row,
)
```

### `callUniformSelect`

Selects a subset of points from a generated A–Ci curve and assigns limitation labels.

```python
Data, t, accepted, random_indices = callUniformSelect(
    LeafMassFlux,
    LeafState,
    Input,
    Photosynthesis,
    requiredpoints,
    Data,
    t,
    rng=rng,
    min_curve_ids=2,
    min_selected_ids=2,
)
```

Important arguments:

| Argument | Description |
|---|---|
| `requiredpoints` | Number of points to select from the curve. |
| `rng` | NumPy random generator used for reproducible point selection. |
| `min_curve_ids` | Minimum number of limitation classes required in the full curve. |
| `min_selected_ids` | Minimum number of limitation classes required after point selection. |

### `generate_one_valid_curve`

Generates one valid synthetic A–Ci curve.

```python
curve_data = generate_one_valid_curve(
    seed=42,
    ci_min=20.0,
    ci_max=1000.0,
    n_ci_points=100,
)
```

The returned object contains:

```python
{
    "ci": ...,
    "aNet": ...,
    "ID": ...,
    "paramArray": ...
}
```

### `dataGeneratorUniform_forGNN`

Generates multiple synthetic A–Ci curves sequentially.

```python
Data = dataGeneratorUniform_forGNN(
    Number_of_curves=6000,
    output_dir="GeneratedData/GNN_Data",
    seed=42,
)
```

### `dataGeneratorUniform_forGNN_parallel`

Generates multiple synthetic A–Ci curves in parallel.

This is the recommended function for large datasets.

```python
Data = dataGeneratorUniform_forGNN_parallel(
    Number_of_curves=6000,
    output_dir="GeneratedData/GNN_Data",
    seed=42,
    n_jobs=-1,
)
```

Here:

```python
n_jobs=-1
```

uses all available CPU cores.

To use a fixed number of CPU cores:

```python
Data = dataGeneratorUniform_forGNN_parallel(
    Number_of_curves=6000,
    output_dir="GeneratedData/GNN_Data",
    seed=42,
    n_jobs=4,
)
```

## Installation

Install the required Python packages:

```bash
pip install numpy pandas scipy joblib jupyter
```

For Google Colab, most required packages are usually preinstalled. If needed, install missing packages using:

```python
!pip install joblib
```

## Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the main notebook and run the cells in order.

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

## Returned Data Structure

The generator returns a Python dictionary:

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
- `Data["paramArray"]` contains parameters used to generate each curve.

Because the number of selected points may vary between curves, the outputs are stored as ragged lists rather than a single rectangular array.

## Reproducibility

The generator supports reproducible data generation through a random seed.

```python
seed=42
```

For parallel generation, independent child seeds are created using:

```python
np.random.SeedSequence(seed)
```

This keeps the full run reproducible while giving each worker an independent random stream.

## Parallel Execution

Parallel execution is handled using `joblib`.

```python
from joblib import Parallel, delayed
```

The parallel version generates complete A–Ci curves independently across workers. This is efficient because each curve is independent of the others.

Recommended function:

```python
dataGeneratorUniform_forGNN_parallel(...)
```

## Example Use Case for GNNs

Each generated A–Ci curve can be converted into a graph:

- nodes: selected measurement points,
- node features: `Ci`, `A_net`, and optionally auxiliary curve-derived signals,
- node labels: limitation-state IDs,
- edges: k-nearest-neighbor or auxiliary-signal-guided connections.

This enables node-level classification of photosynthetic limitation states using GNN models such as:

- Graph Convolutional Networks (GCN),
- Graph Attention Networks (GAT),
- Graph U-Net,
- custom GNN architectures for A–Ci curve analysis.

## License

Add the appropriate license for your project, such as MIT, Apache-2.0, or a custom academic-use license.

## Citation / Acknowledgement

If you use this code in a publication, thesis, or research project, please cite or acknowledge the repository appropriately.
