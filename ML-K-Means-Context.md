# ML K-Means House Grouping — Implementation Context and Execution Specification

## 1. Document purpose

This file is a complete implementation handoff for the 4Geeks Academy **K-Means Project Tutorial**. It explains the project’s purpose, translates the short tutorial into an unambiguous execution plan, defines the data and model contracts, identifies important machine-learning risks, specifies expected artifacts, and gives an implementation agent a step-by-step validation checklist.

This is a planning and execution specification only. No dataset, notebook, clustering model, classifier, or application has been implemented by the author of this file.

Primary source:

- [4Geeks Academy K-Means tutorial](https://github.com/4GeeksAcademy/k-means-project-tutorial/blob/main/README.md)
- [Tutorial repository](https://github.com/4GeeksAcademy/k-means-project-tutorial)
- [Tutorial dataset](https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv)
- [4Geeks machine-learning project template](https://github.com/4GeeksAcademy/machine-learning-python-template)

Relevant implementation references:

- [scikit-learn `KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn `StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [scikit-learn `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)

## 2. Project purpose

Build a reproducible machine-learning workflow that groups California census block groups according to geographic location and median income, then trains a supervised classifier to reproduce those machine-generated group assignments.

The tutorial intentionally combines two learning paradigms:

1. **Unsupervised learning:** K-Means discovers six clusters without any human-provided target labels.
2. **Pseudo-label generation:** the fitted K-Means model assigns a cluster identifier to each training and test row.
3. **Supervised learning:** a classification model learns to predict the K-Means cluster identifier from the same three input features.

The educational objective is not to predict house value. It is to demonstrate how unlabeled observations can first be segmented by an unsupervised model and then used as labeled examples for a supervised model.

## 3. Problem definition

### 3.1 Unit of observation

Each row represents a California census block group from the 1990 census. A row is not an individual house. User-facing explanations must use terms such as **census block group**, **housing area**, or **record**, and should avoid claiming that each row is a single home.

### 3.2 Input features

Only these columns are in scope, in this exact order:

| Position | Column | Meaning | Expected type |
|---:|---|---|---|
| 1 | `Latitude` | Geographic latitude of the block group | numeric floating point |
| 2 | `Longitude` | Geographic longitude of the block group | numeric floating point |
| 3 | `MedInc` | Median income for the block group, in the dataset’s documented units | numeric floating point |

The feature order is part of the model contract. Every fit, predict, save/reload check, and inference example must preserve it.

### 3.3 Unsupervised output

- Algorithm: scikit-learn `KMeans`.
- Fixed cluster count: `n_clusters=6`, as required by the tutorial.
- Output for each row: one integer cluster identifier in the set `{0, 1, 2, 3, 4, 5}`.
- Working column name: `cluster`.

Cluster identifiers are nominal labels, not ranked quantities. Cluster `5` is not inherently larger, wealthier, or better than cluster `1`. K-Means label numbers can also be permuted between independently fitted models even if the underlying groupings are similar. Any semantic description must come from calculated cluster profiles rather than from the numeric label.

### 3.4 Supervised output

- Task type: six-class classification.
- Inputs: `Latitude`, `Longitude`, and `MedInc`.
- Target: the `cluster` label produced by the already-fitted K-Means workflow.
- Evaluation question: how faithfully can the classifier reproduce the K-Means assignments on held-out observations?

This evaluation is **agreement with K-Means**, not accuracy against real-world ground truth. The project has no human-validated region or socioeconomic class labels. The final report must state this limitation plainly.

## 4. Required outcomes

The completed implementation must:

1. Start from a repository based on the 4Geeks machine-learning template, or adapt the current repository to its conventions.
2. Load `housing.csv` from the tutorial URL or a local raw-data copy.
3. Validate that the three required features exist and are usable.
4. Split rows reproducibly into training and test sets before fitting preprocessing or models.
5. Fit all preprocessing only on training data.
6. Train a six-cluster K-Means model using the training set only.
7. Add K-Means-generated `cluster` labels to separate training and test data frames.
8. Visualize the training clusters and overlay or otherwise distinguish test-set predictions.
9. Describe the observed geographic and income patterns without inventing unsupported meanings.
10. Compare reasonable supervised classifiers using training data only.
11. Evaluate the selected classifier on the held-out test set with appropriate multiclass metrics.
12. Save the complete K-Means inference workflow and the selected supervised workflow in `models/`.
13. Reload both saved artifacts and prove that their predictions match the pre-save predictions.
14. Preserve a clean, reproducible execution path in `src/app.py`; exploratory work may live in `src/explore.ipynb`.
15. Document commands, assumptions, results, artifact paths, and limitations in the project README.

The tutorial does not provide a minimum score threshold. The implementation must report actual results and must not invent an accuracy requirement.

## 5. Scope boundaries

### 5.1 In scope

- Dataset acquisition and validation.
- Focused exploratory analysis of the three required columns.
- Reproducible train/test splitting.
- Leakage-safe scaling.
- Six-cluster K-Means fitting and prediction.
- Cluster profiling and visualization.
- Supervised candidate comparison and selection.
- Multiclass evaluation.
- Model persistence and reload verification.
- Focused automated tests for critical data and prediction contracts.
- Reproducibility documentation.

### 5.2 Out of scope unless separately requested

- Predicting `MedHouseVal` or any other housing-value target.
- Geocoding, external census enrichment, or reverse mapping coordinates to city names.
- A web application, API, dashboard, or deployment.
- A database or authentication system.
- Production monitoring or cloud infrastructure.
- Replacing K-Means with DBSCAN, hierarchical clustering, Gaussian mixtures, or another clustering algorithm.
- Changing the required cluster count from six merely because another value scores better.
- Treating generated cluster labels as verified social, economic, demographic, or policy categories.
- Extensive or costly hyperparameter searches.

## 6. Important implementation decisions and tutorial ambiguities

### 6.1 Scale the three features before K-Means

K-Means minimizes Euclidean distances. `Latitude`, `Longitude`, and `MedInc` have different units and numerical spreads; without scaling, the feature with the largest effective spread can dominate the distance calculation. Therefore:

- fit `StandardScaler` on `X_train` only;
- transform `X_train` and `X_test` with that fitted scaler;
- fit K-Means on scaled training features;
- include the scaler and K-Means estimator in one saved scikit-learn `Pipeline` so inference cannot omit preprocessing.

This is a necessary implementation detail even though the tutorial does not explicitly mention scaling.

### 6.2 Split before fitting anything

The tutorial requires train and test sets so the trained unsupervised model can assign clusters to new points. Use a deterministic random split before scaling or clustering. A default implementation contract is:

- `test_size=0.20`;
- `random_state=42`;
- `shuffle=True`.

No target exists at this stage, so the original train/test split cannot be stratified by `cluster`. Do not fit K-Means on the full dataset and then split; that would allow the held-out observations to influence scaling statistics and centroids.

### 6.3 Six clusters are a requirement, not a value to optimize away

The tutorial explicitly requires six clusters. Inertia or silhouette scores may be reported as diagnostics, but they must not be used to silently substitute a different `n_clusters`. If an optional elbow or silhouette study is included, present it as analysis and still deliver the required six-cluster model.

### 6.4 Use deterministic K-Means settings

Specify all behavior that affects reproducibility rather than relying on version-dependent defaults. Recommended explicit settings are:

- `n_clusters=6`;
- `init="k-means++"`;
- `n_init=10`;
- `random_state=42`;
- retain documented defaults for `max_iter`, `tol`, and algorithm unless results justify a change.

Using an explicit integer for `n_init` avoids a behavior change across scikit-learn versions where the default became `"auto"`.

### 6.5 Keep cluster values integer for modeling

The tutorial notes that the new cluster column may need to be categorized. Interpret this carefully:

- keep the modeling target as integer class labels `0` through `5`;
- optionally cast a copy to pandas `category` for summaries or plot legends;
- do not treat cluster numbers as continuous regression targets;
- do not one-hot encode `y` unless a chosen estimator explicitly requires it; normal scikit-learn classifiers accept a one-dimensional multiclass target.

### 6.6 Make the required plot unambiguous

A two-dimensional dot plot cannot simultaneously place all three features on axes. Use this required primary visualization:

- x-axis: `Longitude`;
- y-axis: `Latitude`;
- color: `cluster`;
- training points: small, partially transparent circles;
- test points: a distinct marker such as `x`, or hollow points with a dark edge;
- legend/colorbar: clearly maps all six cluster IDs;
- title and caption: explain whether the figure shows training assignments, test predictions, or both.

Show `MedInc` in at least one companion analysis: a per-cluster box plot, violin plot, summary table, faceted scatter plots, or an optional 3D plot. Do not encode so many properties into one chart that it becomes unreadable.

### 6.7 Treat the supervised stage as model imitation

The test pseudo-labels come from `kmeans_pipeline.predict(X_test)`. They are valid targets for measuring how well the supervised model imitates K-Means, but they do not independently validate whether the clusters are meaningful. Report supervised metrics as **K-Means label agreement** or **fidelity**.

### 6.8 Save preprocessing with each model that needs it

The saved K-Means artifact must be the full scaler-plus-K-Means pipeline, not a bare estimator. The selected supervised artifact should likewise be a full inference pipeline whenever its estimator requires preprocessing. An implementation that saves a model but requires an undocumented, separately fitted scaler is incomplete.

## 7. Recommended repository layout

Adapt to existing project conventions without refactoring unrelated files:

```text
project-root/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── housing.csv
│   └── processed/
│       ├── housing_train_clustered.csv
│       └── housing_test_clustered.csv
├── models/
│   ├── kmeans_pipeline.joblib
│   ├── supervised_classifier.joblib
│   └── model_metadata.json
├── reports/
│   └── figures/
│       ├── train_clusters.png
│       ├── train_test_clusters.png
│       ├── income_by_cluster.png
│       └── supervised_confusion_matrix.png
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   └── utils.py
└── tests/
    └── test_workflow.py
```

The notebook is for investigation and narrative. `src/app.py` is the authoritative clean execution path. Do not make the final result depend on manually running notebook cells in a particular order.

## 8. Dependencies and environment

The 4Geeks template currently includes the principal packages needed for this work. The implementation should verify rather than assume availability of:

- Python 3.11 or the version defined by the template/dev container;
- `pandas` for tabular data;
- `numpy` for numerical checks;
- `scikit-learn` for splitting, preprocessing, clustering, classification, metrics, and pipelines;
- `matplotlib` and `seaborn` for plots;
- `joblib` for local trusted-model persistence;
- Jupyter only if `src/explore.ipynb` will be used;
- `pytest` if automated tests are added and it is not already available.

Implementation steps:

1. Inspect `requirements.txt`, the dev-container configuration, and the current Python version.
2. Reuse the project’s dependency mechanism; do not introduce a second package manager.
3. Add only genuinely missing direct dependencies.
4. Record exact runtime versions in `models/model_metadata.json` or a generated environment report.
5. From the project root, install with the template’s documented command, normally `pip install -r requirements.txt` for local work.
6. Confirm imports before beginning analysis.

## 9. Detailed implementation sequence

### Phase 0 — Establish the execution contract

1. Read the project README, `requirements.txt`, `.gitignore`, `src/`, `data/`, and `models/` before editing.
2. Preserve existing user work and repository conventions.
3. Define one shared constant for the feature order:
   `FEATURES = ["Latitude", "Longitude", "MedInc"]`.
4. Define one shared seed, recommended as `RANDOM_STATE = 42`.
5. Define artifact paths relative to the project root, not the current shell directory.
6. Decide whether the source of truth is the remote CSV URL or `data/raw/housing.csv`. Prefer retaining a local raw copy for reproducibility while documenting the canonical URL.
7. Ensure generated artifacts, caches, and notebook checkpoints are handled intentionally by `.gitignore`. Do not ignore required source files or documentation.

Exit condition: the agent can state the feature order, split, seed, model paths, and authoritative run command before modeling begins.

### Phase 1 — Acquire and load the dataset

Canonical URL:

```text
https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv
```

Execution steps:

1. If network access is available, download the file once to `data/raw/housing.csv` or load it with pandas and save the immutable raw copy.
2. If network access is unavailable, require the user or environment to place the tutorial file at that path; do not silently substitute a different housing dataset.
3. Load with `pandas.read_csv`.
4. Record the source URL, retrieval date, file size, row count, column count, and full column list.
5. Do not mutate or overwrite the raw CSV.
6. Fail with a useful error if the file is missing, empty, malformed, or not parseable as CSV.

Exit condition: a pandas data frame exists and its provenance is recorded.

### Phase 2 — Validate and understand the data

Perform and report all of the following before modeling:

1. Display or log `shape`, column names, data types, and the first several rows.
2. Confirm that `Latitude`, `Longitude`, and `MedInc` exist with exact capitalization.
3. Select only those columns into a new data frame; do not accidentally include a housing-value column or other numeric fields.
4. Coerce the three fields to numeric only if conversion is safe and explicitly checked. Values that become `NaN` must be reported.
5. Count missing and infinite values per feature.
6. Count exact duplicate feature rows. Do not automatically delete duplicates unless the analysis establishes that they are erroneous; census records may legitimately share values.
7. Calculate `count`, mean, standard deviation, minimum, quartiles, and maximum for every feature.
8. Check physically plausible coordinate ranges and flag, rather than silently repair, obviously impossible latitude or longitude values.
9. Inspect feature distributions with histograms and/or box plots.
10. Record the cleaning policy. A reasonable default for this tutorial is to fail clearly if required values are missing, because silently dropping rows changes the data; if rows are dropped, report exact counts and reasons.
11. Preserve original row indices or add a stable `row_id` so labels can be rejoined without misalignment.

Exit condition: the three-feature modeling frame is numeric, finite, nonempty, and traceable to source rows.

### Phase 3 — Create the holdout split

1. Split the validated three-feature frame into `X_train` and `X_test` using the agreed `test_size=0.20`, `random_state=42`, and shuffling.
2. Do not pass a `stratify` argument because cluster targets do not yet exist.
3. Preserve pandas data frames and their indices.
4. Assert that train and test indices are disjoint.
5. Assert that their union contains every retained source row exactly once.
6. Record split sizes and seed.
7. Do not inspect test-set model performance while selecting K-Means settings or supervised hyperparameters.

Exit condition: an immutable training partition and a held-out test partition exist with no overlap.

### Phase 4 — Build and fit the K-Means inference pipeline

Construct a scikit-learn pipeline with these ordered steps:

1. `StandardScaler`.
2. `KMeans(n_clusters=6, init="k-means++", n_init=10, random_state=42)`.

Then:

1. Fit the pipeline on `X_train` only.
2. Generate `train_clusters = pipeline.predict(X_train)`.
3. Generate `test_clusters = pipeline.predict(X_test)` without refitting.
4. Verify both arrays are one-dimensional, integer-like, aligned with their input rows, and contain no values outside `0..5`.
5. Confirm all six labels appear in training. If a label is absent, treat this as an unexpected failure and inspect the pipeline or data.
6. Record K-Means convergence information such as `n_iter_` and inertia from the fitted K-Means step.
7. Extract cluster centers in scaled space for technical diagnostics.
8. Convert a copy of the scaled centers back to original feature units with the fitted scaler’s `inverse_transform`; use these original-unit centers in human-readable reporting.
9. Do not call `fit_predict` separately on train and test, and never fit a second K-Means model on test data.

Exit condition: one training-fitted pipeline labels both partitions and produces six valid training clusters.

### Phase 5 — Attach labels safely

1. Create `train_labeled = X_train.copy()` and `test_labeled = X_test.copy()`.
2. Add `cluster` by positional assignment only after confirming lengths match.
3. Preserve original indices or stable row IDs.
4. Keep `cluster` as an integer target in modeling frames.
5. Optionally create a categorical plotting column without replacing the integer target.
6. Save sorted, clearly named processed CSV files if processed artifacts are part of the repository contract.
7. On CSV reload, confirm the feature and target dtypes remain valid.

Avoid assigning labels directly to a sliced source frame in a way that risks pandas `SettingWithCopyWarning` or index misalignment.

Exit condition: separate train and test frames each contain the three features plus a correctly aligned `cluster` column.

### Phase 6 — Profile and interpret the clusters

Create a cluster profile table using training data only. For each cluster, report:

- number of rows and percentage of training rows;
- mean and median `Latitude`;
- mean and median `Longitude`;
- mean and median `MedInc`;
- the inverse-transformed centroid in original units;
- optionally standard deviations and interquartile ranges.

Interpretation rules:

1. Describe patterns literally: for example, relatively northern/southern, coastal/inland only when supported by coordinates, and lower/higher median income relative to other clusters.
2. Do not assign city, county, demographic, risk, desirability, or policy labels without external evidence.
3. State that cluster sizes need not be equal.
4. State that centroids are mathematical centers and may not correspond to an actual observation.
5. If clusters overlap visually in geographic space, check whether `MedInc` separates them before declaring the clustering unsuccessful.

Exit condition: every cluster has a quantitative profile and a cautious written interpretation.

### Phase 7 — Produce the required visual analysis

Create at minimum:

1. **Training cluster map**
   - longitude on x;
   - latitude on y;
   - color by training cluster;
   - readable legend or discrete colorbar;
   - equal or otherwise sensible geographic aspect;
   - transparent points to manage overplotting.

2. **Training plus test prediction map**
   - retain the same cluster-to-color mapping as the training plot;
   - show test rows with a clearly different marker or outline;
   - label the two partitions in the legend;
   - do not recolor cluster IDs between plots.

3. **Income-by-cluster view**
   - box plot, violin plot, or cluster summary plot of `MedInc`;
   - make outliers visible without allowing them to compress the entire view beyond usefulness.

4. **Optional diagnostic view**
   - a 3D scatter of longitude, latitude, and income; or
   - multiple two-dimensional feature-pair plots; or
   - a silhouette plot for the required six-cluster solution.

Every figure must have a descriptive title, axis labels, cluster legend/colorbar, and a caption or nearby markdown interpretation. Save deterministic figures under `reports/figures/` using non-interactive rendering in the script.

The success question for the test overlay is not whether test points duplicate the training plot. It is whether new points receive plausible labels consistent with nearby training regions in the three-dimensional feature space, with no preprocessing or label-mapping mismatch.

Exit condition: plots distinguish train from test, use consistent colors, and show the contribution of income as well as geography.

### Phase 8 — Select a supervised classifier responsibly

The tutorial asks the implementer to study which classifier is most useful. Do not choose a model without evidence. Use `X_train` and `train_clusters` only for selection.

Recommended small candidate set:

1. **Multinomial-capable logistic regression in a scaling pipeline**
   - strong, fast linear baseline;
   - conceptually suitable because nearest-centroid decision regions have linear boundaries between centroid pairs in scaled space.

2. **Decision tree classifier**
   - interpretable nonlinear baseline;
   - can overfit and creates axis-aligned approximations of cluster boundaries.

3. **Random forest classifier**
   - robust nonlinear candidate;
   - usually needs little preprocessing but is less interpretable and larger on disk.

Selection protocol:

1. Use stratified cross-validation on the training pseudo-labels, or create a stratified validation split from the training partition.
2. Keep the final test set untouched during model choice.
3. Use the same folds for every candidate.
4. Compare at least accuracy and macro-averaged F1. Macro F1 prevents large clusters from completely hiding weak performance on small clusters.
5. Record mean and variability across folds, training time if useful, interpretability, and artifact complexity.
6. If tuning, use a small documented parameter grid and run it only within training data.
7. Select the simplest model whose validation performance and per-class behavior are adequate; do not claim that complexity is automatically better.
8. Refit the chosen full supervised pipeline on all `X_train` and `train_clusters` after selection.

The supervised classifier should accept a raw three-column frame in the same feature order used elsewhere. If it needs scaling, embed scaling in its pipeline.

Exit condition: model selection is evidence-based, reproducible, and isolated from the final test set.

### Phase 9 — Evaluate the selected supervised model once on test data

1. Predict `supervised_test_predictions = classifier.predict(X_test)`.
2. Compare against `test_clusters`, which were produced by the training-fitted K-Means pipeline.
3. Report:
   - overall accuracy;
   - balanced accuracy;
   - macro and weighted precision, recall, and F1;
   - a classification report for labels `0..5`;
   - a 6×6 confusion matrix with explicit label order;
   - per-cluster support.
4. Plot the confusion matrix with readable labels.
5. Confirm predictions contain no unknown classes.
6. Identify which K-Means clusters are easiest and hardest for the classifier to imitate.
7. If a cluster has no test support, report it; do not conceal undefined metrics.
8. Do not repeatedly tune against these test results.

Interpretation must use language such as: “The classifier reproduced K-Means assignments with X accuracy on the holdout set.” It must not say that the classifier discovered true California housing regions.

Exit condition: the selected classifier has a complete, honest test report and confusion matrix.

### Phase 10 — Persist and verify artifacts

Required model artifacts:

- `models/kmeans_pipeline.joblib`: fitted scaler plus K-Means model.
- `models/supervised_classifier.joblib`: fitted complete supervised inference workflow.

Recommended metadata artifact:

- `models/model_metadata.json` containing:
  - creation timestamp;
  - dataset URL and, if calculated, checksum;
  - retained row count;
  - exact feature names and order;
  - train/test split parameters;
  - K-Means parameters;
  - supervised estimator class and relevant parameters;
  - cluster profile summary or path to it;
  - validation and final test metrics;
  - Python, scikit-learn, pandas, NumPy, and joblib versions;
  - artifact filenames.

Persistence steps:

1. Create `models/` if missing.
2. Save both fitted workflows with `joblib.dump`.
3. Load them into fresh variables with `joblib.load`.
4. Predict the same fixed sample with both original and reloaded objects.
5. Assert exact equality of the before/after cluster predictions and classifier predictions.
6. Assert the reloaded objects accept a pandas frame with the three named features.
7. Document that joblib is pickle-based and model files must only be loaded from trusted sources.
8. Document that compatible Python and dependency versions are required; saved scikit-learn artifacts are not a substitute for keeping the training code and environment metadata.

Exit condition: both artifacts reload successfully and reproduce predictions exactly.

### Phase 11 — Create the clean execution path

Move clean, deterministic workflow code into `src/app.py`. It should execute in this order:

1. resolve project paths;
2. load and validate raw data;
3. select features;
4. split data;
5. fit the K-Means pipeline;
6. generate train/test cluster labels;
7. produce profiles and figures;
8. select or instantiate the documented supervised model;
9. train and evaluate it;
10. save artifacts and metadata;
11. reload and verify artifacts;
12. print a concise result summary and artifact locations.

If candidate comparison is computationally separated from the final run, preserve the selection evidence in the notebook or a report and hard-code only the justified final configuration in `app.py`.

The script must be runnable from the project root with the template convention:

```bash
python src/app.py
```

Avoid absolute machine-specific paths, hidden notebook state, interactive-only plotting calls, and network downloads on every run when a valid raw copy already exists.

Exit condition: a new environment can reproduce the workflow by following the README and running the documented command.

### Phase 12 — Document the completed experiment

The final README should include:

1. project purpose in plain language;
2. data source and unit of observation;
3. exact three-feature contract;
4. environment setup and run commands;
5. data split, scaling, and leakage-prevention choices;
6. K-Means configuration and reason for six clusters;
7. cluster profile table and written observations;
8. training/test visualizations;
9. supervised candidate comparison and selection rationale;
10. final holdout metrics and confusion matrix;
11. artifact paths and safe loading instructions;
12. reproducibility settings and dependency versions;
13. limitations, especially pseudo-label fidelity versus ground truth;
14. delivery instructions required by 4Geeks: commit, push to the learner’s repository, and submit the repository link through 4Geeks.com. Those external actions require the user’s explicit authorization and credentials.

## 10. Suggested internal interfaces

Names may follow project conventions, but responsibilities should remain separated and testable:

```text
load_housing_data(source) -> DataFrame
validate_and_select_features(frame, features) -> DataFrame
split_features(X, test_size, random_state) -> (X_train, X_test)
build_kmeans_pipeline(random_state) -> Pipeline
assign_clusters(pipeline, X_train, X_test) -> (train_labels, test_labels)
build_cluster_profiles(X_train, train_labels, pipeline) -> DataFrame
compare_classifiers(X_train, train_labels, random_state) -> results
evaluate_classifier(model, X_test, test_labels) -> metrics
save_and_verify_artifacts(kmeans_pipeline, classifier, paths, sample) -> None
```

Interface rules:

- raw feature frames use the exact named columns in the exact agreed order;
- validation functions fail clearly rather than silently changing the schema;
- fit operations are explicit and occur only on training data;
- plotting functions receive labels rather than refitting models;
- evaluation returns structured values in addition to printing them;
- filesystem paths are `pathlib.Path` objects resolved from the project root.

## 11. Testing and validation plan

Testing should focus on observable contracts and regression risks, using a small synthetic frame where possible so tests remain fast.

### 11.1 Data contract tests

- Valid numeric input with the three required columns is accepted.
- A missing required column produces a clear error naming the column.
- Non-numeric, missing, or infinite required values follow the documented failure/cleaning policy.
- Extra source columns are excluded from model input.
- Feature order is normalized to `Latitude`, `Longitude`, `MedInc`.
- Empty input is rejected.

### 11.2 Split tests

- Fixed seed produces repeatable train/test indices.
- Train and test indices do not overlap.
- The union of partitions equals the retained input rows.
- Split sizes match the documented policy.

### 11.3 K-Means workflow tests

- The pipeline contains preprocessing followed by K-Means.
- The scaler and K-Means are fit on training data only.
- `n_clusters` is exactly six.
- Predictions have one label per row and all labels are within `0..5`.
- A repeated fit with the same data and seed produces identical assignments.
- Test prediction does not mutate or refit the pipeline.
- Inverse-transformed centroids have shape `(6, 3)` and the expected feature order.

For a very small synthetic test fixture, ensure there are enough distinct samples for six clusters or use a fixture specifically designed with six separated groups.

### 11.4 Label alignment tests

- The `cluster` column length equals its feature frame length.
- Original indices remain associated with the correct predictions.
- Saving and reloading processed CSVs does not silently turn targets into invalid values.

### 11.5 Supervised workflow tests

- Candidate selection never receives the final test target.
- The chosen classifier can fit all observed training classes.
- Predictions are restricted to known cluster IDs.
- Metric calculation supplies an explicit label order of `0..5`.
- The confusion matrix has shape `(6, 6)` even when a small fixture lacks a predicted class.

### 11.6 Persistence tests

- Both expected model files are created.
- Both files load successfully in the same environment.
- Reloaded predictions exactly match predictions made before saving.
- The K-Means artifact includes preprocessing.
- A raw three-column pandas frame can be passed directly to each reloaded workflow.

### 11.7 End-to-end smoke test

Run the workflow on a small deterministic subset or fixture and confirm it produces:

- two labeled frames;
- six-cluster predictions where the fixture supports them;
- the required figures;
- structured metrics;
- both model artifacts;
- metadata;
- successful reload verification.

### 11.8 Validation commands

Adapt to the repository’s actual tooling. A typical sequence is:

```bash
pytest -q
python src/app.py
```

If configured, also run the project’s formatter/linter and any type checker. Never claim a command passed unless it was actually run.

## 12. Acceptance criteria

The implementation is complete only when all applicable statements are true:

### Data and reproducibility

- [ ] The canonical tutorial dataset is used and its provenance is documented.
- [ ] Only `Latitude`, `Longitude`, and `MedInc` are model inputs.
- [ ] Missing, infinite, non-numeric, and duplicate data have been inspected and reported.
- [ ] Train and test rows are disjoint and reproducibly split.
- [ ] The random seed and exact feature order are documented.

### K-Means

- [ ] Scaling is fit on training data only.
- [ ] K-Means is fit on training data only.
- [ ] The model uses exactly six clusters.
- [ ] K-Means randomness and initialization behavior are explicit.
- [ ] Both training and test rows receive cluster labels from the same fitted workflow.
- [ ] `cluster` is stored as a categorical class label, not a regression quantity.
- [ ] Original-unit centroids and cluster sizes are reported.

### Analysis and plots

- [ ] A geographic training scatter plot is present.
- [ ] Test predictions are visibly distinguished on a consistent cluster map.
- [ ] Median income is analyzed by cluster.
- [ ] Every cluster has a quantitative profile.
- [ ] Written interpretation avoids unsupported semantic claims.

### Supervised learning

- [ ] At least a baseline and one meaningful alternative were compared, or a documented constraint justifies a narrower study.
- [ ] Selection uses training data only.
- [ ] Final evaluation uses the held-out test partition.
- [ ] Accuracy, macro F1, balanced accuracy, per-class results, and a confusion matrix are reported.
- [ ] Results are described as fidelity to K-Means pseudo-labels, not truth-label accuracy.

### Persistence and execution

- [ ] The complete K-Means pipeline is saved in `models/`.
- [ ] The complete supervised inference workflow is saved in `models/`.
- [ ] Both artifacts reload and reproduce predictions exactly.
- [ ] Model/environment metadata is recorded.
- [ ] `python src/app.py` is a clean documented run path.
- [ ] Focused automated tests pass, or exact environmental limitations are reported.

## 13. Common failure modes to avoid

1. **Fitting on all rows before splitting.** This leaks test distribution information into preprocessing and centroids.
2. **Fitting a separate K-Means model to test data.** Cluster label numbers would not have a stable shared meaning and the test set would no longer represent new points.
3. **Skipping scaling.** Euclidean distance would weight features according to raw numerical scale rather than the intended equal standardized footing.
4. **Scaling train and test independently.** Test values must use training means and standard deviations.
5. **Saving only bare K-Means.** Inference would be wrong if the same scaler were not reapplied.
6. **Changing `n_clusters` after an elbow study.** Six is an explicit exercise requirement.
7. **Treating cluster IDs as ordered.** They are nominal and arbitrarily numbered.
8. **Using test results for classifier selection.** This turns the final holdout into a tuning set.
9. **Reporting only accuracy.** Unequal cluster sizes can hide weak minority-cluster imitation.
10. **Calling pseudo-label accuracy real-world accuracy.** No ground-truth region labels exist in the exercise.
11. **Plotting only longitude and latitude, then ignoring income.** The model clusters in three dimensions; two points that overlap geographically may differ in income.
12. **Changing colors between plots.** A cluster must retain the same visual identity across every figure in one run.
13. **Losing index alignment while adding labels.** Predictions must be attached to the exact rows from which they were generated.
14. **Depending on notebook execution state.** The final workflow must run cleanly from the script.
15. **Loading untrusted joblib files.** Joblib relies on pickle semantics and can execute malicious content during loading.
16. **Hard-coding local absolute paths.** Artifacts and data paths must resolve portably from the repository root.

## 14. Expected final narrative

The completed project report should answer these questions with measured evidence:

1. What does one dataset row represent?
2. Why were these three features selected?
3. Why was scaling necessary for K-Means?
4. How were leakage and reproducibility controlled?
5. Where are the six clusters located in latitude/longitude space?
6. How does median income differ among them?
7. How many training and test rows belong to each cluster?
8. Do test points visually receive assignments consistent with the fitted training pattern?
9. Which supervised classifiers were compared, using which selection metric?
10. How closely did the selected classifier reproduce K-Means on held-out rows?
11. Which clusters were most often confused?
12. Where are the two saved workflows, and were reload predictions verified?
13. What can and cannot be concluded from these generated labels?

## 15. Implementation handoff order

The next agent should execute the work in this exact dependency order:

1. Inspect repository and environment.
2. Acquire and validate the canonical CSV.
3. Select and lock the feature schema.
4. Create the reproducible holdout split.
5. Fit the scaler-plus-K-Means pipeline on training data.
6. Generate aligned train/test labels.
7. Profile and visualize the clusters.
8. Compare supervised classifiers on training data only.
9. Refit the chosen supervised workflow on all training rows.
10. Evaluate once against K-Means test labels.
11. Persist both complete workflows and metadata.
12. Reload and verify predictions.
13. Clean the script and notebook.
14. Add and run focused tests.
15. Finish the README with actual results and limitations.

Do not skip directly to classifier training: every later stage depends on a single, leakage-safe K-Means labeling contract.

