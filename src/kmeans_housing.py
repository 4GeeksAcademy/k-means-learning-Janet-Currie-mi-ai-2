"""Leakage-safe K-Means grouping and supervised imitation for housing areas."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["Latitude", "Longitude", "MedInc"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CLUSTERS = 6
CLASS_LABELS = list(range(N_CLUSTERS))
DATASET_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
)
SIMPLICITY_ORDER = ("logistic_regression", "decision_tree", "random_forest")
CLUSTER_COLORS = {
    0: "#4C78A8",
    1: "#F58518",
    2: "#54A24B",
    3: "#E45756",
    4: "#72B7B2",
    5: "#B279A2",
}


def artifact_paths(root: Path | str) -> dict[str, Path]:
    root_path = Path(root).resolve()
    figures = root_path / "reports" / "figures"
    models = root_path / "models"
    return {
        "root": root_path,
        "data": root_path / "housing.csv",
        "train_csv": root_path / "data" / "processed" / "housing_train_clustered.csv",
        "test_csv": root_path / "data" / "processed" / "housing_test_clustered.csv",
        "models": models,
        "kmeans": models / "kmeans_pipeline.joblib",
        "supervised": models / "supervised_classifier.joblib",
        "metadata": models / "model_metadata.json",
        "figures": figures,
        "train_plot": figures / "train_clusters.png",
        "train_test_plot": figures / "train_test_clusters.png",
        "income_plot": figures / "income_by_cluster.png",
        "confusion_plot": figures / "supervised_confusion_matrix.png",
    }


def load_housing_data(source: Path | str) -> pd.DataFrame:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Housing data not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Housing data is empty: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV at {path}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"Housing data has no rows: {path}")
    return frame


def validate_and_select_features(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...] = FEATURES,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if frame.shape[0] == 0:
        raise ValueError("Input frame is empty")

    required = list(features)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    selected = frame.loc[:, required].copy()
    for column in required:
        converted = pd.to_numeric(selected[column], errors="coerce")
        invalid = converted.isna() | np.isinf(converted)
        if bool(invalid.any()):
            raise ValueError(
                f"Column {column} has {int(invalid.sum())} missing, non-numeric, or infinite values"
            )
        selected[column] = converted.astype(float)

    if selected.index.name is None:
        selected.index.name = "row_id"
    return selected


def coordinate_flag_count(frame: pd.DataFrame) -> int:
    latitude = frame["Latitude"]
    longitude = frame["Longitude"]
    invalid = (latitude < -90) | (latitude > 90) | (longitude < -180) | (longitude > 180)
    return int(invalid.sum())


def summarize_features(frame: pd.DataFrame) -> dict:
    summary = frame.describe().to_dict()
    return {
        "row_count": int(frame.shape[0]),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "duplicate_feature_rows": int(frame.duplicated().sum()),
        "coordinate_flags": coordinate_flag_count(frame),
        "summary": summary,
    }


def split_features(
    features: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not features.index.is_unique:
        raise ValueError("Feature index values must be unique before splitting")

    train_frame, test_frame = train_test_split(
        features,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    overlap = train_frame.index.intersection(test_frame.index)
    if len(overlap) > 0:
        raise AssertionError("Train and test indices overlap")
    combined = train_frame.index.union(test_frame.index)
    if len(combined) != len(features.index) or not combined.equals(features.index):
        # RangeIndex.union sort order can differ from the original order.
        if set(combined) != set(features.index) or len(train_frame) + len(test_frame) != len(features):
            raise AssertionError("Train and test indices do not cover every row exactly once")
    return train_frame, test_frame


def build_kmeans_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=N_CLUSTERS,
                    init="k-means++",
                    n_init=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def assign_clusters(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_labels = np.asarray(pipeline.predict(x_train)).astype(int)
    test_labels = np.asarray(pipeline.predict(x_test)).astype(int)
    _validate_label_array(train_labels, len(x_train), require_all=True)
    _validate_label_array(test_labels, len(x_test), require_all=False)
    return train_labels, test_labels


def _validate_label_array(labels: np.ndarray, expected_length: int, require_all: bool) -> None:
    if labels.ndim != 1 or len(labels) != expected_length:
        raise ValueError("Cluster predictions must contain one label per row")
    unexpected = sorted(set(labels.tolist()) - set(CLASS_LABELS))
    if unexpected:
        raise ValueError(f"Cluster predictions contain values outside 0..5: {unexpected}")
    if require_all:
        missing = sorted(set(CLASS_LABELS) - set(labels.tolist()))
        if missing:
            raise RuntimeError(
                f"Training predictions are missing cluster ids {missing}. "
                "The fitted six-cluster workflow did not use every cluster."
            )


def attach_clusters(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    label_array = np.asarray(labels).astype(int)
    if len(label_array) != len(features):
        raise ValueError("Cluster labels must match the feature frame length")
    labeled = features.copy()
    labeled["cluster"] = label_array
    return labeled


def original_unit_centers(pipeline: Pipeline) -> np.ndarray:
    scaler = pipeline.named_steps["scaler"]
    kmeans = pipeline.named_steps["kmeans"]
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    if centers.shape != (N_CLUSTERS, len(FEATURES)):
        raise ValueError(
            f"Expected inverse-transformed centers with shape {(N_CLUSTERS, len(FEATURES))}, "
            f"got {centers.shape}"
        )
    return centers


def build_cluster_profiles(
    x_train: pd.DataFrame,
    train_labels: np.ndarray,
    pipeline: Pipeline,
) -> pd.DataFrame:
    labels = np.asarray(train_labels).astype(int)
    centers = original_unit_centers(pipeline)
    rows = []
    total = len(x_train)
    for cluster_id in CLASS_LABELS:
        part = x_train.iloc[np.flatnonzero(labels == cluster_id)]
        centroid = centers[cluster_id]
        rows.append(
            {
                "cluster": cluster_id,
                "count": int(len(part)),
                "percent": float(100.0 * len(part) / total) if total else 0.0,
                "latitude_mean": float(part["Latitude"].mean()) if len(part) else np.nan,
                "latitude_median": float(part["Latitude"].median()) if len(part) else np.nan,
                "longitude_mean": float(part["Longitude"].mean()) if len(part) else np.nan,
                "longitude_median": float(part["Longitude"].median()) if len(part) else np.nan,
                "medinc_mean": float(part["MedInc"].mean()) if len(part) else np.nan,
                "medinc_median": float(part["MedInc"].median()) if len(part) else np.nan,
                "centroid_latitude": float(centroid[0]),
                "centroid_longitude": float(centroid[1]),
                "centroid_medinc": float(centroid[2]),
            }
        )
    return pd.DataFrame(rows)


def describe_clusters(x_train: pd.DataFrame, profiles: pd.DataFrame) -> list[str]:
    latitude_cutoff = float(x_train["Latitude"].median())
    longitude_cutoff = float(x_train["Longitude"].median())
    income_cutoff = float(x_train["MedInc"].median())
    statements = [
        "Cluster ids are nominal labels, so a higher number is not a higher rank.",
        "Cluster sizes do not need to be equal.",
        "Centroids are mathematical centers and may not match an actual census block group.",
        "MedInc stays in the file's own units. Values in this dataset are capped near 15.",
        "In this California dataset, farther west is toward the coast and farther east is inland.",
    ]
    for row in profiles.itertuples(index=False):
        latitude_phrase = _relative_phrase(
            row.latitude_median, latitude_cutoff, "farther north", "farther south"
        )
        longitude_phrase = _relative_phrase(
            row.longitude_median, longitude_cutoff, "farther east", "farther west"
        )
        income_phrase = _relative_phrase(
            row.medinc_median, income_cutoff, "higher median income", "lower median income"
        )
        statements.append(
            f"Cluster {int(row.cluster)} has {int(row.count)} training rows "
            f"({row.percent:.1f}%). It sits {latitude_phrase} and {longitude_phrase}, "
            f"with {income_phrase} relative to the training medians."
        )
    return statements


def _relative_phrase(value: float, cutoff: float, higher: str, lower: str) -> str:
    if value > cutoff:
        return higher
    if value < cutoff:
        return lower
    return "near the training median"


def choose_n_splits(labels: np.ndarray, requested: int = 5) -> int:
    counts = pd.Series(np.asarray(labels)).value_counts()
    smallest = int(counts.min())
    n_splits = min(requested, smallest)
    if n_splits < 2:
        raise ValueError(
            "Need at least 2 rows in every class for cross-validation; "
            f"smallest class has {smallest}"
        )
    return n_splits


def build_supervised_pipeline(name: str, random_state: int = RANDOM_STATE) -> Pipeline:
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "decision_tree":
        return Pipeline(
            [
                ("classifier", DecisionTreeClassifier(random_state=random_state)),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(random_state=random_state),
                ),
            ]
        )
    raise ValueError(f"Unknown classifier: {name}")


def compare_classifiers(
    x_train: pd.DataFrame,
    train_labels: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> dict:
    labels = np.asarray(train_labels).astype(int)
    n_splits = choose_n_splits(labels, requested=5)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    candidates = {}
    for name in SIMPLICITY_ORDER:
        estimator = build_supervised_pipeline(name, random_state=random_state)
        scores = cross_validate(
            estimator,
            x_train,
            labels,
            cv=splitter,
            scoring=["accuracy", "f1_macro"],
        )
        candidates[name] = {
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"])),
            "macro_f1_mean": float(np.mean(scores["test_f1_macro"])),
            "macro_f1_std": float(np.std(scores["test_f1_macro"])),
            "fit_time_mean": float(np.mean(scores["fit_time"])),
        }

    winner = SIMPLICITY_ORDER[0]
    best_score = candidates[winner]["macro_f1_mean"]
    for name in SIMPLICITY_ORDER[1:]:
        score = candidates[name]["macro_f1_mean"]
        if score > best_score:
            winner = name
            best_score = score

    fitted = build_supervised_pipeline(winner, random_state=random_state)
    fitted.fit(x_train, labels)
    return {
        "n_splits": n_splits,
        "candidates": candidates,
        "winner": winner,
        "fitted_pipeline": fitted,
    }


def evaluate_classifier(
    model,
    x_test: pd.DataFrame,
    test_labels: np.ndarray,
) -> dict:
    observed = np.asarray(test_labels).astype(int)
    predicted = np.asarray(model.predict(x_test)).astype(int)
    unexpected = sorted(set(predicted.tolist()) - set(CLASS_LABELS))
    if unexpected:
        raise ValueError(f"Classifier predicted unknown classes: {unexpected}")

    report = classification_report(
        observed,
        predicted,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(observed, predicted, labels=CLASS_LABELS)
    per_class = []
    for cluster_id in CLASS_LABELS:
        row = report[str(cluster_id)]
        per_class.append(
            {
                "cluster": cluster_id,
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1-score"]),
                "support": int(row["support"]),
            }
        )
    supported = [row for row in per_class if row["support"] > 0]
    easiest = max(supported, key=lambda row: row["f1"]) if supported else None
    hardest = min(supported, key=lambda row: row["f1"]) if supported else None
    return {
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "macro_precision": float(
            precision_score(observed, predicted, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(observed, predicted, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(observed, predicted, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(
                observed, predicted, labels=CLASS_LABELS, average="weighted", zero_division=0
            )
        ),
        "weighted_recall": float(
            recall_score(observed, predicted, labels=CLASS_LABELS, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(observed, predicted, labels=CLASS_LABELS, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": per_class,
        "easiest_cluster": None if easiest is None else int(easiest["cluster"]),
        "hardest_cluster": None if hardest is None else int(hardest["cluster"]),
        "missing_test_support": [row["cluster"] for row in per_class if row["support"] == 0],
    }


def save_cluster_map(
    x_train: pd.DataFrame,
    train_labels: np.ndarray,
    path: Path,
    x_test: pd.DataFrame | None = None,
    test_labels: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_partition(ax, x_train, train_labels, marker="o", size=12, alpha=0.35)
    handles = _cluster_handles(marker="o", prefix="Train")
    title = "Training cluster assignments"
    caption = "Circles are training census block groups. Color is the nominal K-Means cluster id."
    if x_test is not None and test_labels is not None:
        _scatter_partition(ax, x_test, test_labels, marker="x", size=28, alpha=0.9)
        handles.extend(_cluster_handles(marker="x", prefix="Test"))
        title = "Training assignments and test predictions"
        caption = (
            "Circles are training rows and x markers are test predictions. "
            "A cluster keeps the same color in both partitions."
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    latitudes = x_train["Latitude"].to_numpy()
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(latitudes)))))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    _finish_figure(fig, path, caption)


def save_income_plot(x_train: pd.DataFrame, train_labels: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(train_labels).astype(int)
    income = x_train["MedInc"].to_numpy()
    groups = [income[labels == cluster_id] for cluster_id in CLASS_LABELS]
    fig, ax = plt.subplots(figsize=(9, 6))
    box = ax.boxplot(groups, patch_artist=True)
    for patch, cluster_id in zip(box["boxes"], CLASS_LABELS):
        patch.set_facecolor(CLUSTER_COLORS[cluster_id])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, N_CLUSTERS + 1))
    ax.set_xticklabels([str(cluster_id) for cluster_id in CLASS_LABELS])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("MedInc (dataset units)")
    ax.set_title("Median income by training cluster")
    _finish_figure(
        fig,
        path,
        "MedInc stays in file units and is capped near 15. Cluster ids are nominal.",
    )


def save_confusion_plot(matrix: list[list[int]] | np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks(CLASS_LABELS)
    ax.set_yticks(CLASS_LABELS)
    ax.set_xlabel("Classifier prediction")
    ax.set_ylabel("K-Means cluster")
    ax.set_title("Holdout agreement with K-Means labels")
    for row_id in CLASS_LABELS:
        for column_id in CLASS_LABELS:
            ax.text(column_id, row_id, str(int(values[row_id, column_id])), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _finish_figure(
        fig,
        path,
        "Counts measure fidelity to K-Means pseudo-labels, not ground-truth regions.",
    )


def save_labeled_frames(train_labeled: pd.DataFrame, test_labeled: pd.DataFrame, paths: dict[str, Path]) -> None:
    paths["train_csv"].parent.mkdir(parents=True, exist_ok=True)
    train_labeled.sort_index().to_csv(paths["train_csv"], index=True)
    test_labeled.sort_index().to_csv(paths["test_csv"], index=True)


def save_and_verify_artifacts(kmeans_pipeline, classifier, paths: dict[str, Path], sample: pd.DataFrame) -> None:
    if not isinstance(sample, pd.DataFrame):
        raise TypeError("Verification sample must be a pandas DataFrame")
    missing = [column for column in FEATURES if column not in sample.columns]
    if missing:
        raise ValueError(f"Verification sample is missing column(s): {', '.join(missing)}")

    named_sample = sample.loc[:, FEATURES]
    before_clusters = np.asarray(kmeans_pipeline.predict(named_sample))
    before_classes = np.asarray(classifier.predict(named_sample))

    paths["models"].mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_pipeline, paths["kmeans"])
    joblib.dump(classifier, paths["supervised"])

    loaded_kmeans = joblib.load(paths["kmeans"])
    loaded_classifier = joblib.load(paths["supervised"])
    if not any(isinstance(step, StandardScaler) for _, step in loaded_kmeans.steps):
        raise AssertionError("Reloaded K-Means artifact does not include preprocessing")

    after_clusters = np.asarray(loaded_kmeans.predict(named_sample))
    after_classes = np.asarray(loaded_classifier.predict(named_sample))
    if not np.array_equal(before_clusters, after_clusters):
        raise AssertionError("Reloaded K-Means predictions differ from pre-save predictions")
    if not np.array_equal(before_classes, after_classes):
        raise AssertionError("Reloaded classifier predictions differ from pre-save predictions")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def package_versions() -> dict[str, str]:
    import sklearn

    return {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def run_workflow(
    root: Path | str,
    source: Path | str | None = None,
    verbose: bool = False,
) -> dict:
    paths = artifact_paths(root)
    data_path = Path(source) if source is not None else paths["data"]

    def log(message: str) -> None:
        if verbose:
            print(message)

    log(f"Loading {data_path}")
    raw = load_housing_data(data_path)
    features = validate_and_select_features(raw, FEATURES)
    diagnostics = summarize_features(features)
    log(
        f"Rows: {diagnostics['row_count']}; "
        f"duplicate three-feature rows kept: {diagnostics['duplicate_feature_rows']}; "
        f"coordinate flags: {diagnostics['coordinate_flags']}"
    )

    x_train, x_test = split_features(features, TEST_SIZE, RANDOM_STATE)
    log(f"Train rows: {len(x_train)}; test rows: {len(x_test)}")

    kmeans_pipeline = build_kmeans_pipeline(RANDOM_STATE)
    kmeans_pipeline.fit(x_train)
    train_labels, test_labels = assign_clusters(kmeans_pipeline, x_train, x_test)
    kmeans_step = kmeans_pipeline.named_steps["kmeans"]

    train_labeled = attach_clusters(x_train, train_labels)
    test_labeled = attach_clusters(x_test, test_labels)
    save_labeled_frames(train_labeled, test_labeled, paths)

    profiles = build_cluster_profiles(x_train, train_labels, kmeans_pipeline)
    interpretations = describe_clusters(x_train, profiles)
    save_cluster_map(x_train, train_labels, paths["train_plot"])
    save_cluster_map(
        x_train,
        train_labels,
        paths["train_test_plot"],
        x_test=x_test,
        test_labels=test_labels,
    )
    save_income_plot(x_train, train_labels, paths["income_plot"])

    log("Comparing supervised classifiers on training labels")
    comparison = compare_classifiers(x_train, train_labels, RANDOM_STATE)
    classifier = comparison["fitted_pipeline"]
    evaluation = evaluate_classifier(classifier, x_test, test_labels)
    save_confusion_plot(evaluation["confusion_matrix"], paths["confusion_plot"])

    sample = pd.concat([x_train.iloc[:25], x_test.iloc[:25]], axis=0)
    save_and_verify_artifacts(kmeans_pipeline, classifier, paths, sample)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_url": DATASET_URL,
        "dataset_path": str(data_path),
        "dataset_sha256": file_sha256(data_path),
        "file_size_bytes": data_path.stat().st_size,
        "raw_column_list": list(raw.columns),
        "retained_row_count": int(len(features)),
        "feature_names": FEATURES,
        "duplicate_feature_rows_kept": diagnostics["duplicate_feature_rows"],
        "coordinate_flags": diagnostics["coordinate_flags"],
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "shuffle": True,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "kmeans": {
            "n_clusters": N_CLUSTERS,
            "init": "k-means++",
            "n_init": 10,
            "random_state": RANDOM_STATE,
            "n_iter": int(kmeans_step.n_iter_),
            "inertia": float(kmeans_step.inertia_),
        },
        "cluster_profiles": profiles.to_dict(orient="records"),
        "cluster_interpretations": interpretations,
        "supervised_selection": {
            "n_splits": comparison["n_splits"],
            "winner": comparison["winner"],
            "candidates": comparison["candidates"],
            "rule": (
                "Highest mean macro F1 on stratified training folds. "
                "Ties keep the simpler model: logistic regression, decision tree, then random forest."
            ),
        },
        "holdout_fidelity": {
            key: value
            for key, value in evaluation.items()
            if key != "classification_report"
        },
        "classification_report": evaluation["classification_report"],
        "versions": package_versions(),
        "artifacts": {
            "kmeans_pipeline": paths["kmeans"].name,
            "supervised_classifier": paths["supervised"].name,
            "metadata": paths["metadata"].name,
        },
        "reload_verification": "passed",
        "limitation": (
            "Holdout metrics measure agreement with K-Means pseudo-labels. "
            "They are not accuracy against human-labeled regions."
        ),
    }
    write_metadata(metadata, paths["metadata"])
    log(f"Selected classifier: {comparison['winner']}")
    log(
        "Holdout K-Means label agreement: "
        f"accuracy {evaluation['accuracy']:.4f}, macro F1 {evaluation['macro_f1']:.4f}"
    )
    log("Reload verification: passed")
    return {
        "diagnostics": diagnostics,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "profiles": profiles,
        "interpretations": interpretations,
        "kmeans_n_iter": int(kmeans_step.n_iter_),
        "kmeans_inertia": float(kmeans_step.inertia_),
        "comparison": {
            "n_splits": comparison["n_splits"],
            "winner": comparison["winner"],
            "candidates": comparison["candidates"],
        },
        "evaluation": evaluation,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _finish_figure(fig, path: Path, caption: str) -> None:
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _scatter_partition(ax, features: pd.DataFrame, labels: np.ndarray, marker: str, size: int, alpha: float) -> None:
    label_array = np.asarray(labels).astype(int)
    longitude = features["Longitude"].to_numpy()
    latitude = features["Latitude"].to_numpy()
    for cluster_id in CLASS_LABELS:
        mask = label_array == cluster_id
        ax.scatter(
            longitude[mask],
            latitude[mask],
            s=size,
            alpha=alpha,
            c=CLUSTER_COLORS[cluster_id],
            marker=marker,
            linewidths=0.6,
        )


def _cluster_handles(marker: str, prefix: str) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color=CLUSTER_COLORS[cluster_id],
            label=f"{prefix} {cluster_id}",
            linestyle="",
        )
        for cluster_id in CLASS_LABELS
    ]
