# MLDS Assignment 3

Federated EMNIST classification with FedAvg, Ray Core Actors, and Laplace input noise.

The final clean model reaches **80.8% test accuracy**. The four-client Ray run reaches **76.9%**, with weights and validation curves exactly matching its serial reference. The selected noise scale, **b=0.05**, reaches **80.0% test accuracy**. Selection uses validation results, not test accuracy.

## Run

Use Python 3.14. Get `Assignment3-data.zip` from Canvas and put it in `reports/`. The course data are excluded from Git.

From the repository root:

```bash
python -m pip install -r requirements.txt
python reports/run_experiments.py all
python reports/build_report.py
```

Alternatively, open `reports/hw3.ipynb` with the same Python environment, use `reports/` as its working directory, and run all cells. The last cell exports the figures, results, and PDF.

The full run compares input representations, learning rates, round budgets, client participation, and local epochs. It also runs four Ray clients per round, five noise scales with three seeds each, and the validation experiments below. Test data are opened after the model settings and noise choice have been saved.

Individual stages can be run with `python reports/run_experiments.py STAGE`, where `STAGE` is `tune`, `select`, `ray`, `noise`, `validate`, or `test`. Run them in that order. Then run `python reports/make_figures.py` and rebuild the PDF. The `all` command performs these experiment/figure steps in order.

Each run saves its configuration, environment, full learning curves, and timing in `reports/runs/`. Local model checkpoints are ignored by Git. A cached run is reused only when its checkpoint exists and the algorithm source/configuration match. A fresh clone retrains the models. Ray Actor creation makes the full run slower than the serial experiments.

## Files

- `reports/hw3.ipynb`: executable notebook, with outputs.
- `reports/federated.py`: model, client training, sample-weighted FedAvg, evaluation, and Ray Actors.
- `reports/run_experiments.py`: tuning, matched Ray runs, noise comparisons, controls, and final test evaluation.
- `reports/make_figures.py`: plots and the combined `experiment_results.json`.
- `reports/build_report.py`: PDF builder; numbers come from exported JSON.
- `reports/Assignment3_Report.pdf`: submission report.
- `reports/*_results.json`, `selected_config.json`, and `reports/runs/*.json`: recorded numerical evidence.
- `reports/figures/`: report figures.

## Validation experiments

| Experiment | What it checks |
| --- | --- |
| Original pixels vs. `1 - pixels`, matched seeds | Whether the input representation affects optimization |
| Learning-rate and round-budget comparisons | Whether the first run was trained well enough |
| C=5%/10%, E=1/3/5 | Participation and local-work effects at a fixed round budget |
| Within-client shuffled training labels | Whether accuracy depends on the image-label relationship; label histograms stay fixed |
| Fit 16 real images | Whether the network and optimizer can solve a small known task |
| Full serial/Ray match | Whether changing execution preserves weights and learning curves |
| One-CPU Ray cluster | Whether four selected clients finish in four one-Actor batches |
| Injected Actor failure after a completed batch | Whether the round leaves the global model unchanged, releases CPUs, and can be retried correctly |
| Three seeds per noise scale | Training/noise variability on the fixed client split |
| Fresh noise on the same training images | Whether a model depends on the stored perturbations |
| b=3 with the selected and a smaller rate | Numerical failure, rejected updates, and utility under extreme noise |
| Unequal client counts, one client, lr=0 | FedAvg equation limits |
| Zero/invalid noise scales and Laplace moments | Exact recovery, input validation, and the specified distribution |
| Per-client majority baseline and class recall | Whether aggregate accuracy hides weak clients or classes |

The fixed split uses seed 42. The main run uses seed 42; noise comparisons also use seeds 43 and 44. A positive noise scale is selected by validation results: take the largest tested scale whose mean validation accuracy is within two percentage points of b=0.

Training uses CPU with one PyTorch thread per process. The main Ray cluster reserves four CPUs; the resource/failure experiment reserves one. Each Actor reserves one CPU and zero GPUs. An intentional Actor failure produces an expected Ray error message; the validation stage catches it and verifies recovery. The b=3 run also records its expected numerical failure rather than hiding it.

The measured environment is Python 3.14.9, PyTorch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2, Ray 2.60.0, and ReportLab 4.4.9 on Apple M5. Package versions are pinned in `requirements.txt`. Timings depend on the computer.

Laplace perturbation follows the assignment mechanism. A numerical privacy guarantee would also require a defined protected unit, adjacency relation, sensitivity, and release policy. Labels are unchanged.
