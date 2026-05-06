# Multi-SpaCE Counterfactual Dashboard

This dashboard is a lightweight Streamlit app for inspecting already-generated counterfactual explanation files.

## Run

From the repository root:

```bash
streamlit run dashboards/cf_dashboard.py
```

If Streamlit is not installed in the active environment:

```bash
pip install streamlit plotly pandas numpy openpyxl
```

`openpyxl` is only needed when reading `.xlsx` files.

## What It Loads

The dashboard discovers likely result files under a selected root, defaulting to:

```text
experiments/results
```

Supported files:

- `.pickle`, `.pkl`
- `.joblib`
- `.npz`, `.npy`
- `.json`
- `.csv`
- `.xlsx`
- `.parquet`

For legacy Multi-SpaCE `counterfactuals.pickle` files, the saved object is often a list of dictionaries with keys such as `time`, `cfs`, and `fitness_evolution`. Those older files usually do not contain `x_orig` or `x_nun`. When possible, the dashboard infers the dataset and `X_test_indexes` from the result path and nearby `params.json`, then loads `x_orig` and `y_test` from `experiments/data/UCR/<dataset>/`.

Newer result files that include `x_orig`, `nun`, labels, predictions, masks, or metric dictionaries are read directly.

## Features

- Select a result file discovered from disk or paste a path manually.
- Select an explained original instance.
- Inspect metadata, labels, predictions, validity, and candidate count.
- Plot a 3D Pareto/front scatter with selectable x/y/z objective axes.
- Select a candidate manually by `candidate_id`.
- Select a candidate automatically with weighted utility over numeric metrics.
- Inspect candidate metrics in a table.
- Overlay `x_orig`, `x_cf`, and `x_nun` when available.
- Infer and highlight changed regions from `x_orig != x_cf` when no mask is saved.
- Handle univariate and multivariate time series with channel selection.

The app is intentionally file-based and introspective. If a file cannot be parsed, it should show the parsing error in the dashboard instead of crashing silently.
