# Tech Updates

The local environment resolved `requirements.txt` to Python 3.13.14, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, and joblib 1.6.0.

Pandas 3 rejects assigning a string into a float column. Tests that need a non-numeric value build an object column instead of assigning into a float series.

scikit-learn 1.9 logistic regression is used without the removed `multi_class` argument. `solver="lbfgs"` and `max_iter=1000` are explicit.
