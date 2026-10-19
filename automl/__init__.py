"""Unknown-dataset AutoML for the Introduction to Machine Learning hackathon.

Public API::

    from automl import run, solve
    run("data/practice/clf_telco_churn", "outputs/clf_telco_churn")  # files in, files out
    preds, profile = solve(train_df, test_df, {"task": "classification", "target": "y"})
"""
__version__ = "1.0.0"

from .pipeline import run, solve, solve_dataset  # noqa: E402

predict = run

__all__ = ["run", "solve", "solve_dataset", "predict", "__version__"]
