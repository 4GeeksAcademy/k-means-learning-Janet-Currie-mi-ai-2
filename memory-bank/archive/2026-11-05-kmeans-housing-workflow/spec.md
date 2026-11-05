# Spec

The completed workflow must:

- Use the local root `housing.csv` and record the tutorial URL.
- Model only `Latitude`, `Longitude`, and `MedInc`, in that order.
- Split 80/20 with seed 42 before fitting.
- Fit a scaler and six-cluster K-Means pipeline on training rows only.
- Store `cluster` as an integer nominal label.
- Compare logistic regression, a decision tree, and a random forest on training labels only.
- Select the highest mean macro F1, with ties going to the simpler model.
- Report holdout metrics as K-Means agreement.
- Save both pipelines, metadata, labeled CSVs, and four figures.
- Reload the pipelines and match predictions.
- Expose `python src/app.py` and focused pytest contracts.
