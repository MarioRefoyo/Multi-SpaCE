from __future__ import annotations

from pathlib import Path


DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary',
    'Cricket',
    'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    'PEMS-SF',
    'RacketSports', 'SelfRegulationSCP1'
]

MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_noscaling"
METHOD_NAME = "mascots_scalar_gpu"
RESULTS_ROOT = Path("experiments/results")


def find_available_runs(
    dataset: str,
    model_name: str,
    method_name: str,
    results_root: Path,
) -> list[Path]:
    model_results_dir = results_root / dataset / model_name
    if not model_results_dir.is_dir():
        return []

    matches = []
    for run_dir in sorted(model_results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if method_name.lower() not in run_dir.name.lower():
            continue
        if (run_dir / "counterfactuals.pickle").is_file():
            matches.append(run_dir)
    return matches


def main() -> None:
    print(f"Method name filter: {METHOD_NAME}")
    print(f"Model to explain: {MODEL_TO_EXPLAIN_EXPERIMENT_NAME}")
    print(f"Results root: {RESULTS_ROOT}")
    print("")

    total_found = 0
    for dataset in DATASETS:
        matches = find_available_runs(
            dataset=dataset,
            model_name=MODEL_TO_EXPLAIN_EXPERIMENT_NAME,
            method_name=METHOD_NAME,
            results_root=RESULTS_ROOT,
        )

        if matches:
            total_found += len(matches)
            match_names = ", ".join(run_dir.name for run_dir in matches)
            print(f"[FOUND] {dataset}: {match_names}")
        else:
            print(f"[MISSING] {dataset}: no matching run with counterfactuals.pickle")

    print("")
    print(f"Total matching runs found: {total_found}")


if __name__ == "__main__":
    main()
