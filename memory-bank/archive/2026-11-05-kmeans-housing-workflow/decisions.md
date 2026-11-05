# Decisions

- Read `housing.csv` from the repository root. Leave `data/raw/` unchanged.
- Keep five duplicate three-feature rows and all physically flagged coordinates. This file had zero coordinate flags.
- Leave `MedInc` in file units.
- Store cluster ids as integers.
- Leave `src/utils.py` in place and remove the database call from `src/app.py`.
- Put joblib pipelines in the existing `models/` directory.
- Update `README.md` only.
- Choose the classifier by training macro F1, with the simplicity tie break logistic regression, decision tree, then random forest.
- Use 5 stratified folds, reduced only when a class has fewer than 5 rows.
- Use scikit-learn's current multinomial logistic regression (`lbfgs`, `max_iter=1000`) and record the installed version.
- Leave `.gitignore` unchanged so generated project artifacts can be committed later.
