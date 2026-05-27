"""Temporary diagnostic: find the scikit-learn version that pickled the model.

Run from the project root. Delete this file afterwards.
"""
import joblib
import sklearn

print("=" * 60)
print("Installed scikit-learn :", sklearn.__version__)
print("=" * 60)

bundle = joblib.load("models/classifiers.joblib")
print("Bundle keys:", list(bundle.keys()))

for key, est in bundle.items():
    v = getattr(est, "_sklearn_version", "unknown")
    print(f"  {key:>10s}: {type(est).__name__}, pickled with sklearn {v}")

    # Dig into the nested pipeline to confirm the SimpleImputer's version too.
    inner = getattr(est, "estimator", None) or getattr(est, "base_estimator", None)
    if inner is not None and hasattr(inner, "steps"):
        for name, step in inner.steps:
            sv = getattr(step, "_sklearn_version", "unknown")
            print(f"             step '{name}': {type(step).__name__}, sklearn {sv}")

print("=" * 60)
