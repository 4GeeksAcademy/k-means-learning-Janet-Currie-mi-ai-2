"""Run the housing K-Means workflow from the repository root."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kmeans_housing import run_workflow


def main() -> None:
    summary = run_workflow(ROOT, verbose=True)
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    diagnostics = summary["diagnostics"]
    comparison = summary["comparison"]
    evaluation = summary["evaluation"]

    print("Feature order: Latitude, Longitude, MedInc")
    print(f"Rows kept: {diagnostics['row_count']}")
    print(f"Duplicate three-feature rows kept: {diagnostics['duplicate_feature_rows']}")
    print(f"Coordinate flags: {diagnostics['coordinate_flags']}")
    print(f"Train rows: {summary['train_rows']}")
    print(f"Test rows: {summary['test_rows']}")
    print(f"K-Means iterations: {summary['kmeans_n_iter']}")
    print(f"K-Means inertia: {summary['kmeans_inertia']:.4f}")
    print("Cluster profiles:")
    print(summary["profiles"].to_string(index=False))
    print("Cluster notes:")
    for statement in summary["interpretations"]:
        print(f"- {statement}")
    print(f"Cross-validation folds: {comparison['n_splits']}")
    for name, scores in comparison["candidates"].items():
        print(
            f"{name}: mean accuracy {scores['accuracy_mean']:.4f} "
            f"(std {scores['accuracy_std']:.4f}), "
            f"mean macro F1 {scores['macro_f1_mean']:.4f} "
            f"(std {scores['macro_f1_std']:.4f})"
        )
    print(f"Selected classifier: {comparison['winner']}")
    print(
        "Holdout K-Means label agreement: "
        f"accuracy {evaluation['accuracy']:.4f}, "
        f"balanced accuracy {evaluation['balanced_accuracy']:.4f}, "
        f"macro F1 {evaluation['macro_f1']:.4f}"
    )
    print(f"Easiest cluster to imitate: {evaluation['easiest_cluster']}")
    print(f"Hardest cluster to imitate: {evaluation['hardest_cluster']}")
    print(f"Test clusters with no support: {evaluation['missing_test_support']}")
    print("Artifact paths:")
    for key in ("train_csv", "test_csv", "kmeans", "supervised", "metadata", "figures"):
        print(f"- {key}: {summary['paths'][key]}")
    print("Reload verification: passed")
    print(
        "These scores measure how closely the classifier reproduces K-Means labels. "
        "They are not accuracy against real-world region labels."
    )


if __name__ == "__main__":
    main()
