# Machine-Learning-Guided Temporal Separator Solver

This repository contains the code used to study the **TimeSep** problem in temporal graphs, combining:

- an exact **ILP formulation with contiguity constraints**, and
- a **machine-learning-guided online solver** for larger instances.

The pipeline has three main stages:

1. **Offline dataset construction**  
   Exact ILP solutions are computed and combined with sampled temporal paths to build labeled node-time examples.

2. **Offline model training**  
   A neural scoring model is trained on the joint real + synthetic dataset.

3. **Online inference / solving**  
   The trained model is used to guide an interval-aware search procedure, followed by a repair phase.

## Repository contents

- `pure_ilp_temporal_separator_contiguity.py`  
  Exact ILP solver for the contiguity version of the temporal separator problem.

- `model_inductive_temporal.py`  
  Inductive neural model used to score node-time pairs.

- `offline_build_joint_real_synthetic_dataset_contiguity.py`  
  Builds a unified training dataset from real and synthetic temporal graphs.

- `train_joint_real_synthetic_contiguity.py`  
  Trains the inductive temporal scorer on the generated dataset.

- `online_real_joint_real_synthetic_interval_beam_repair.py`  
  Runs the ML-guided online solver on real-world temporal graph instances.

- `online_synthetic_joint_real_synthetic_interval_beam_repair.py`  
  Runs the ML-guided online solver on synthetic temporal graph instances.

## Input format

The code expects temporal graph files in a UVT-like format, where each non-comment line is:

u v t

representing a directed temporal edge from `u` to `v` at timestamp `t`.

### Real-world files
Real-world files may contain multiple source-target-deadline combinations in the header, for example:

```text
# combo 1: source = 10 target = 25 deadline = 8
# combo 2: source = 11 target = 30 deadline = 12
```

### Synthetic files
Synthetic files contain exactly one instance in the header, for example:

```text
# source = 10
# target = 25
# deadline = 8
# max_timestamp = 50
```

## Main dependencies

- Python 3.10+
- PyTorch
- Gurobi / gurobipy

## How to run

### 1. Build the joint dataset
Run:

```bash
python offline_build_joint_real_synthetic_dataset_contiguity.py
```

This creates:

- `joint_real_synthetic_dataset.pt`
- `meta.json`

inside the output directory specified in the script.

### 2. Train the model
Run:

```bash
python train_joint_real_synthetic_contiguity.py
```

This saves the trained model checkpoint and training metadata.

### 3. Run the online solver on real instances
Run:

```bash
python online_real_joint_real_synthetic_interval_beam_repair.py
```

Before running, edit the `CITY` variable and relevant paths in the script.

### 4. Run the online solver on synthetic instances
Run:

```bash
python online_synthetic_joint_real_synthetic_interval_beam_repair.py
```

Before running, edit the `SYNTHETIC_INPUT` path in the script.

## Important note on paths

The current scripts use **absolute local paths** (for example `/Users/...`) in their setup sections.  
Before running the code on a different machine, these paths must be updated manually.

This repository keeps the code close to the experimental version used in the paper, so the path configuration is intentionally left explicit in each script.

## Notes on the methodology

- The exact ILP solver enforces **contiguity**, meaning each selected vertex may be active only over one contiguous time interval.
- The dataset builder combines:
  - exact ILP positives,
  - soft positives from sampled path frequencies,
  - hard negatives,
  - random negatives.
- The online solver uses:
  - ML-guided interval-aware search,
  - feasibility checking through surviving temporal paths,
  - a repair phase to improve the final separator.

## Output

The online solvers write human-readable result files summarizing:

- instance information,
- runtime,
- separator size before repair,
- separator size after repair,
- interval assignments per selected vertex.

## Citation

If you use this code in academic work, please cite the associated paper.

