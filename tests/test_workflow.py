"""Contract tests for the housing K-Means workflow."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kmeans_housing import (  # noqa: E402
    FEATURES,
    assign_clusters,
    attach_clusters,
    build_kmeans_pipeline,
    build_cluster_profiles,
    choose_n_splits,
    compare_classifiers,
    evaluate_classifier,
    load_housing_data,
    original_unit_centers,
    run_workflow,
    save_and_verify_artifacts,
    split_features,
    validate_and_select_features,
)


def _separated_frame(n_per: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = [
        (33.0, -117.0, 2.0),
        (34.0, -118.0, 5.0),
        (36.0, -120.0, 3.0),
        (37.5, -122.0, 8.0),
        (39.0, -121.0, 4.0),
        (41.0, -123.5, 6.5),
    ]
    frames = []
    for latitude, longitude, income in centers:
        frames.append(
            pd.DataFrame(
                {
                    "MedHouseVal": np.full(n_per, 2.0),
                    "MedInc": income + rng.normal(0, 0.05, n_per),
                    "Longitude": longitude + rng.normal(0, 0.02, n_per),
                    "Latitude": latitude + rng.normal(0, 0.02, n_per),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_valid_frame_selects_features_in_contract_order():
    selected = validate_and_select_features(_separated_frame(n_per=2))
    assert list(selected.columns) == FEATURES
    assert "MedHouseVal" not in selected.columns
    assert selected.shape == (12, 3)


def test_missing_column_names_the_column():
    frame = _separated_frame(n_per=2).drop(columns=["MedInc"])
    with pytest.raises(ValueError, match="MedInc"):
        validate_and_select_features(frame)


def test_empty_input_is_rejected():
    empty = pd.DataFrame(columns=FEATURES)
    with pytest.raises(ValueError, match="empty"):
        validate_and_select_features(empty)


def test_non_numeric_values_are_rejected():
    frame = pd.DataFrame(
        {
            "Latitude": ["abc", "33.0"],
            "Longitude": ["-117.0", "-118.0"],
            "MedInc": ["2.0", "3.0"],
        }
    )
    with pytest.raises(ValueError, match="Latitude"):
        validate_and_select_features(frame)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_missing_and_infinite_values_are_rejected(bad_value):
    frame = _separated_frame(n_per=2)
    frame.loc[0, "Latitude"] = bad_value
    with pytest.raises(ValueError, match="Latitude"):
        validate_and_select_features(frame)


def test_coordinate_flags_do_not_drop_rows():
    frame = validate_and_select_features(_separated_frame(n_per=2))
    frame.loc[frame.index[0], "Latitude"] = 120.0
    assert len(frame) == 12
    from kmeans_housing import coordinate_flag_count

    assert coordinate_flag_count(frame) == 1


def test_missing_file_raises_a_clear_error(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_housing_data(missing)


def test_split_is_repeatable_disjoint_and_complete():
    features = validate_and_select_features(_separated_frame(n_per=10, seed=1))
    train_a, test_a = split_features(features, test_size=0.20, random_state=42)
    train_b, test_b = split_features(features, test_size=0.20, random_state=42)

    assert train_a.index.equals(train_b.index)
    assert test_a.index.equals(test_b.index)
    assert train_a.index.intersection(test_a.index).empty
    assert set(train_a.index).union(set(test_a.index)) == set(features.index)
    assert len(train_a) == 48
    assert len(test_a) == 12
    assert "stratify" not in inspect.signature(split_features).parameters


def test_kmeans_pipeline_contract_and_repeatable_labels():
    features = validate_and_select_features(_separated_frame(n_per=15))
    x_train, x_test = split_features(features)
    first = build_kmeans_pipeline(42)
    second = build_kmeans_pipeline(42)
    assert [name for name, _ in first.steps] == ["scaler", "kmeans"]
    assert first.named_steps["kmeans"].n_clusters == 6

    first.fit(x_train)
    second.fit(x_train)
    centers = first.named_steps["kmeans"].cluster_centers_.copy()
    iterations = first.named_steps["kmeans"].n_iter_
    train_labels, test_labels = assign_clusters(first, x_train, x_test)

    assert np.array_equal(first.predict(x_train), second.predict(x_train))
    assert np.array_equal(centers, first.named_steps["kmeans"].cluster_centers_)
    assert first.named_steps["kmeans"].n_iter_ == iterations
    assert train_labels.shape == (len(x_train),)
    assert test_labels.shape == (len(x_test),)
    assert set(train_labels).issubset(set(range(6)))
    assert set(np.unique(train_labels)) == set(range(6))
    assert original_unit_centers(first).shape == (6, 3)


def test_cluster_labels_stay_aligned_and_reload_as_integers(tmp_path):
    features = validate_and_select_features(_separated_frame(n_per=4))
    shuffled = features.sample(frac=1, random_state=0)
    labels = np.arange(len(shuffled)) % 6
    labeled = attach_clusters(shuffled, labels)

    assert list(labeled.index) == list(shuffled.index)
    assert np.array_equal(labeled["cluster"].to_numpy(), labels)
    assert pd.api.types.is_integer_dtype(labeled["cluster"])

    path = tmp_path / "labeled.csv"
    labeled.to_csv(path, index=True)
    reloaded = pd.read_csv(path, index_col=0)
    assert np.array_equal(reloaded["cluster"].to_numpy(), labels)
    assert pd.api.types.is_integer_dtype(reloaded["cluster"])


def test_classifier_selection_uses_training_labels_only():
    parameters = inspect.signature(compare_classifiers).parameters
    assert "x_test" not in parameters
    assert "test_labels" not in parameters

    features = validate_and_select_features(_separated_frame(n_per=12))
    pipeline = build_kmeans_pipeline(42)
    pipeline.fit(features)
    labels = pipeline.predict(features)
    result = compare_classifiers(features, labels, random_state=42)

    assert result["winner"] in {"logistic_regression", "decision_tree", "random_forest"}
    observed = set(np.unique(labels).tolist())
    fitted_classes = set(result["fitted_pipeline"].named_steps["classifier"].classes_.tolist())
    assert observed.issubset(fitted_classes)


def test_metrics_use_explicit_six_by_six_label_order():
    class ConstantModel:
        def predict(self, features):
            return np.zeros(len(features), dtype=int)

    frame = validate_and_select_features(_separated_frame(n_per=2))
    observed = np.array([0, 1] + [0] * (len(frame) - 2))
    metrics = evaluate_classifier(ConstantModel(), frame, observed)
    matrix = np.asarray(metrics["confusion_matrix"])

    assert matrix.shape == (6, 6)
    assert matrix[0, 0] == len(frame) - 1
    assert matrix[1, 0] == 1
    assert metrics["missing_test_support"] == [2, 3, 4, 5]


def test_choose_n_splits_shrinks_to_the_smallest_class():
    assert choose_n_splits(np.array([0, 0, 0, 1, 1, 1, 1, 1])) == 3
    with pytest.raises(ValueError, match="smallest class"):
        choose_n_splits(np.array([0, 0, 1]))


def test_saved_pipelines_reload_and_keep_the_scaler(tmp_path):
    features = validate_and_select_features(_separated_frame(n_per=12))
    x_train, x_test = split_features(features)
    kmeans_pipeline = build_kmeans_pipeline(42)
    kmeans_pipeline.fit(x_train)
    train_labels, _ = assign_clusters(kmeans_pipeline, x_train, x_test)
    comparison = compare_classifiers(x_train, train_labels, random_state=42)
    paths = {
        "models": tmp_path / "models",
        "kmeans": tmp_path / "models" / "kmeans_pipeline.joblib",
        "supervised": tmp_path / "models" / "supervised_classifier.joblib",
    }
    sample = x_test.iloc[:8]
    save_and_verify_artifacts(
        kmeans_pipeline,
        comparison["fitted_pipeline"],
        paths,
        sample,
    )

    import joblib

    loaded = joblib.load(paths["kmeans"])
    assert isinstance(loaded, Pipeline)
    assert any(isinstance(step, StandardScaler) for _, step in loaded.steps)
    named = pd.DataFrame(sample.loc[:, FEATURES])
    assert np.array_equal(loaded.predict(named), kmeans_pipeline.predict(named))


def test_small_workflow_writes_required_artifacts(tmp_path):
    source = tmp_path / "housing.csv"
    _separated_frame(n_per=20, seed=2).to_csv(source, index=False)
    summary = run_workflow(tmp_path, source=source, verbose=False)

    assert summary["train_rows"] + summary["test_rows"] == 120
    assert Path(summary["paths"]["train_csv"]).exists()
    assert Path(summary["paths"]["test_csv"]).exists()
    assert Path(summary["paths"]["kmeans"]).exists()
    assert Path(summary["paths"]["supervised"]).exists()
    assert Path(summary["paths"]["metadata"]).exists()
    for name in ("train_plot", "train_test_plot", "income_plot", "confusion_plot"):
        assert Path(summary["paths"][name]).exists()
    assert np.asarray(summary["evaluation"]["confusion_matrix"]).shape == (6, 6)
    assert len(summary["profiles"]) == 6
