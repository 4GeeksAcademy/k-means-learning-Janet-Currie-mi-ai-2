# K-Means Housing Workflow

## Purpose

Group California census block groups with a six-cluster K-Means model, then train a classifier that reproduces those generated labels. The exercise demonstrates unsupervised labeling followed by supervised imitation. It does not predict house value.

## Final Behavior

`python src/app.py` loads the root `housing.csv`, keeps all 20,640 rows, and models only `Latitude`, `Longitude`, and `MedInc` in that order. It splits 80/20 with seed 42 before fitting, scales with a training-only `StandardScaler`, and fits `KMeans(n_clusters=6, init="k-means++", n_init=10, random_state=42)` on the training rows. The same pipeline labels the test rows.

A logistic regression pipeline, a decision tree, and a random forest are compared with 5-fold stratified cross-validation on the training labels. The highest mean macro F1 wins. Ties keep the simpler model in this order: logistic regression, decision tree, random forest. The completed run selected random forest. Holdout agreement with K-Means was accuracy 0.9952 and macro F1 0.9934.

Both full inference pipelines are saved under `models/` and reloaded. Reloaded predictions matched the pre-save predictions. Cluster ids are stored as integers and described as nominal classes.

## Architecture and Data Flow

`src/kmeans_housing.py` holds the workflow. `src/app.py` resolves the repository root and calls `run_workflow`. `src/utils.py` remains the unused template database helper. `src/explore.ipynb` calls the same module for loading, clustering, profiles, and figures, then reads `models/model_metadata.json` for the supervised result.

Raw input is the root `housing.csv`. Processed labeled frames go to `data/processed/`. Figures go to `reports/figures/`. The canonical tutorial URL is recorded as provenance and is not downloaded on each run.

## Important Files

- `housing.csv`
- `src/kmeans_housing.py`
- `src/app.py`
- `src/explore.ipynb`
- `tests/test_workflow.py`
- `models/kmeans_pipeline.joblib`
- `models/supervised_classifier.joblib`
- `models/model_metadata.json`
- `data/processed/housing_train_clustered.csv`
- `data/processed/housing_test_clustered.csv`
- `reports/figures/`
- `README.md`

## Interfaces and Contracts

- `FEATURES = ["Latitude", "Longitude", "MedInc"]`
- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.20`
- `N_CLUSTERS = 6`
- Dataset URL: `https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv`
- Public functions include `load_housing_data`, `validate_and_select_features`, `split_features`, `build_kmeans_pipeline`, `assign_clusters`, `build_cluster_profiles`, `compare_classifiers`, `evaluate_classifier`, `save_and_verify_artifacts`, and `run_workflow`.
- Validation rejects missing columns, empty input, and missing, non-numeric, or infinite required values. It keeps duplicate feature rows and physically implausible coordinates, and reports both.
- Classifier cross-validation uses 5 folds unless the smallest training class has fewer rows. The floor is 2.

## Decisions and Constraints

- The root CSV is the raw source. `data/raw/` stays empty.
- Six clusters stay fixed.
- Scaling is inside each pipeline that needs it. The saved K-Means artifact includes the scaler. Logistic regression has its own scaler. The tree and forest do not.
- Logistic regression uses `solver="lbfgs"` and `max_iter=1000`, without the removed `multi_class` argument.
- `MedInc` is not rescaled into dollars before modeling.
- Cluster labels stay integer dtype.
- The database helper is not called.
- Joblib artifacts must be loaded only from a trusted source.

## Validation

On November 5, 2026, in the local `.venv`:

- `pip install -r requirements.txt` completed.
- `pytest -q` passed 16 tests.
- `python src/app.py` completed, printed the split and metrics, and reported reload verification passed.

Recorded runtime: Python 3.13.14, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0.

## Maintenance Notes

Refitting without the same seed can renumber clusters. Compare profiles, not the integer ids, across runs. Do not drop the five duplicate three-feature rows unless a later decision changes the row contract. Do not point the loader at a different housing file. The supervised score is fidelity to K-Means. Adding `MedHouseVal` or changing `n_clusters` changes the exercise.
