# MLDS Assignment 3

Federated EMNIST classification with sample-weighted FedAvg, Ray Core client Actors, and Laplace input perturbation.

## Files

- `reports/hw3.ipynb` contains the implementation, plots, validation experiments, and executed outputs.
- `reports/Assignment3_Report.pdf` contains the report.
- `reports/experiment_results.json` contains numerical results exported by the notebook.
- `reports/figures/` contains the report figures exported by the notebook.
- `reports/build_report.py` rebuilds the PDF from the exported results and figures.

## Run

Obtain `Assignment3-data.zip` from the course Canvas Files page and place it in `reports/`, next to the notebook. The archive is excluded from version control.

```bash
python -m pip install -r requirements.txt
cd reports
```

Open `hw3.ipynb` using that Python environment and run all cells in order with `reports/` as the working directory. The notebook selects CUDA, MPS, or CPU according to availability. Ray Actors each use one CPU and one PyTorch compute thread. The cluster needs at least one CPU. With fewer than four CPUs, selected clients run in batches.

Sections 1-3 prepare data, train local and sequential models, compare hyperparameters, and evaluate the selected model. Section 4 runs 20 rounds with four Ray clients per round. Section 5 runs four noise scales for 50 rounds each. Section 6 adds controlled validation experiments and repeated noise runs. Section 7 exports the results.

The verified environment uses Python 3.14.9, PyTorch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2, and Ray 2.60.0 on an Apple M5 system. Floating-point behavior can vary across devices. Notebook execution takes several minutes, including Actor startup and repeated experiments.

Rebuild the report from the repository root after the notebook finishes.

```bash
python reports/build_report.py
```

## Validation experiments

| Experiment | Expected behavior | Notebook location |
| --- | --- | --- |
| Unequal client aggregation | Match an explicit sample-weighted average | Section 3 |
| Per-client majority baseline | Measure improvement over local class imbalance | Section 3 |
| Split isolation | Zero shared training/validation indices | Section 6 |
| Single-client limit | FedAvg matches local training | Section 6 |
| Zero learning rate | Model parameters stay fixed within rounding tolerance | Section 6 |
| Small real-data fit | Track optimization on 16 fixed examples | Section 6 |
| Matched serial/Ray updates | Same model, clients, and seeds give matching parameters | Section 6 |
| Zero-noise control | Reproduce the matched clean-data experiment | Sections 5-6 |
| Laplace moments | Mean near zero and mean absolute perturbation near b | Section 5 |
| Seed robustness | Repeat each noise scale with three training/noise seeds | Section 6 |
| Strong perturbation | Quantify utility when noise dominates pixel values | Section 6 |

The local 80/20 split stays fixed across runs. Main runs use seed 42. The noise robustness comparison adds seeds 43 and 44. Validation chooses hyperparameters. Held-out test images are used only for evaluation.

The input-perturbation experiments follow the assignment mechanism. A formal privacy guarantee requires a protected unit, adjacency relation, sensitivity calibration, and a definition of released information.
