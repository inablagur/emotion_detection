"""
3_train_shallow_baseline.py — Shallow baselines for emotion classification

Why TF-IDF + linear models here?
- Short texts → high-dimensional and mostly-zero (sparse) feature vectors.
- TF-IDF maps token counts to a weighted, sparse representation that works extremely well with *linear* classifiers.


Models included:
- lr   → Logistic Regression (multinomial, L2): strong on high-dimensional sparse text; probabilistic outputs; interpretable via coefficients; fast; handles multiclass directly; supports class weighting.
- lsvm → LinearSVC (hinge loss, one-vs-rest): very strong baseline for text; efficient on sparse features; interpretable via linear weights.
- cnb  → Complement Naive Bayes: probabilistic model tailored for text; uses complement statistics to stabilize rare features; extremely fast and robust to imbalance in sparse data.


This script:
1) Loads preprocessed CSVs (train/validation/test) from data/.
2) Builds TF-IDF → classifier pipelines.
3) Runs RandomizedSearchCV with macro-F1 on validation.
4) Saves per-model validation JSON reports to reports/ folder.
5) Selects the best model by validation macro-F1 (tie-break accuracy).
6) Retrains the winner on train + validation and evaluates on test.
7) Saves winner test JSON and serialized pipelines to models/.

Outputs:
- models/
    shallow_lr.pkl
    shallow_lsvm.pkl
    shallow_cnb.pkl
    shallow_winner.pkl
- reports/
    shallow_lr_val.json
    shallow_lsvm_val.json
    shallow_cnb_val.json

    shallow_winner_test.json

    shallow_top_features_lr.txt           (optional, only if LR trained)
    shallow_top_features_lsvm.txt         (optional, only if LinearSVC trained)
"""

# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------- Import Packages --------------------------------------------------------------
import argparse
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# --------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------- Utilities -----------------------------------------------------------------
def ensure_dirs(*paths: Path):
    """
    Ensure each given directory exists; create it (and parents) if missing.

    Args:
        *paths (Path): One or more directory paths.

    Returns:
        None
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def timer():
    """
    Starts a high-resolution timer and returns a callable that reports the
    number of seconds elapsed since the timer started.
    *Relevant for training time comparison.

    Args:
        None

    Returns:
        Callable[[], float]: elapsed time in seconds.
    """
    start = time.perf_counter()
    return lambda: time.perf_counter() - start

def latency_ms_per_sample(model, X):
    """
    Measures the average prediction latency per sample in milliseconds.
    *Relevant for inference speed measurement and comparison

    Args:
        model (Pipeline): A trained pipeline with a `.predict()` method.
        X (array-like or Series): Input samples to predict.

    Returns:
        float: Average prediction latency in milliseconds per sample.
    """
    _ = model.predict(X[:5])
    t = timer()
    _ = model.predict(X)
    elapsed = t()
    return (elapsed / len(X)) * 1000.0

def model_file_size_mb(path: Path) -> float:
    """
    Returns the size of a saved model file in megabytes.

    Args:
        path (Path): Path to the model file.

    Returns:
        float: File size in megabytes, rounded to three decimal places.
    """
    return round(path.stat().st_size / (1024 * 1024), 3)

def to_python_types(obj):
    """
    Converts NumPy data types to native Python types for safe JSON serialization.

    Args:
        obj: The object to convert (could be NumPy scalar, NumPy array, or native type).

    Returns:
        Any: A Python-native version of the input object.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a pretty-printed UTF-8 JSON file.

    NumPy data types inside the dictionary are automatically converted
    to Python-native types.

    Args:
        path (Path): Path where the JSON file will be saved.
        data (dict): Data to serialize and save.

    Returns:
        None
    """
    clean = json.loads(json.dumps(data, default=to_python_types))
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

def extract_top_terms_txt(pipeline: Pipeline, labels_order, k=20) -> str:
    """
    Generates a text block with the top-k most influential n-grams per class
    for linear models (e.g. Logistic Regression, LinearSVC etc...)

    The feature names are taken from the TF-IDF vectorizer,
    and the importance scores come from the classifier coefficients.

    Args:
        pipeline (Pipeline): A trained scikit-learn pipeline with 'tfidf' and 'clf' steps.
        labels_order (list): List of class labels in fixed order.
        k (int, optional): Number of top terms to display per class. Default is 20.

    Returns:
        str: A formatted multi-line string with top terms per class.
    """
    # TODO: What does it mean vec: TfidfVectorizer = pipeline.named_steps["tfidf"]? What is the vec variable? Why is it needed? which of them is the variable? I don't understand the syntax here.
    vec: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return ""
    feature_names = np.array(vec.get_feature_names_out())
    lines = []
    lines.append(f"# {clf.__class__.__name__} — top {k} features per class")
    for i, cls in enumerate(labels_order):
        coefs = clf.coef_[i] if clf.coef_.ndim > 1 else clf.coef_.ravel()
        # For one-vs-rest LinearSVC, classes_ gives order; LR multinomial is per class as well.
        # argsort top-k largest positive weights
        topk_idx = np.argsort(coefs)[-k:][::-1]
        terms = feature_names[topk_idx]
        lines.append(f"\n[{cls}]")
        lines.append(", ".join(terms))
    return "\n".join(lines)

def compute_metrics(y_true, y_pred, labels_order):
    """
    Calculates accuracy, macro-F1, per-class F1 scores, and confusion matrix.

    Args:
        y_true (Series or array-like): Ground truth labels.
        y_pred (Series or array-like): Predicted labels.
        labels_order (list): Fixed list of class labels to ensure consistent ordering.

    Returns:
        dict: A dictionary containing:
            - 'accuracy' (float)
            - 'macro_f1' (float)
            - 'per_class_f1' (dict of label: score)
            - 'confusion_matrix' (list of lists, aligned with labels_order)
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, average=None, labels=labels_order)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "per_class_f1": {str(lbl): float(val) for lbl, val in zip(labels_order, per_class)},
        "confusion_matrix": cm.astype(int).tolist()
    }


# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------- Randomized search spaces ---------------------------------------------------------
def get_param_distributions(model_name: str, search_class_weight: bool):
    """
    This funcion defines the hyperparameter search space for a given shallow model.

    The returned dictionary is designed to be passed into RandomizedSearchCV's `param_distributions`
    argument. It defines the search space for each model type.

    Args:
        model_name (str): One of {"lr", "lsvm", "cnb"}. See "Models included" in the script header.
            
        search_class_weight (bool): If True, include 'class_weight'
            in the search space for LR and LinearSVC, trying both None and 'balanced' options. This can help with imbalanced data.  

    Returns:
        dict: Parameter distribution mapping for use in RandomizedSearchCV.
              *Keys match the pipeline parameter names (e.g., 'clf__C').
    """

    # Logistic regression model:
    if model_name == "lr":
        return {
            "clf__C": loguniform(1e-2, 1e2),                                                        # C controls L2 strength
            **({"clf__class_weight": [None, "balanced"]} if search_class_weight else {})            # class_weight only if requested; helpful if imbalance shows up.
        }
    
    # SVM model:
    if model_name == "lsvm":
        return {
            "clf__C": loguniform(1e-2, 1e2),                                                        # C controls regularization strength
            **({"clf__class_weight": [None, "balanced"]} if search_class_weight else {})            # class_weight only if requested; helpful if imbalance shows up.
        }
        
    # Complement Naive Bayes model:
    if model_name == "cnb":
        # CNB handles class imbalance internally
        return {
            "clf__alpha": loguniform(1e-3, 10)                                                      # alpha controls smoothing;
        }
    raise ValueError(f"Unknown model '{model_name}'")

# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------ Main --------------------------------------------------------------------

if __name__ == "__main__":
    
    # Create parser:
    parser = argparse.ArgumentParser(
        description="Train shallow TF-IDF baselines (LR, LinearSVC, ComplementNB) with RandomizedSearchCV."
    )
    # Data
    parser.add_argument("--train", type=Path, default=Path("data/train_clean.csv"), help="Path to cleaned TRAIN split CSV with columns: text, emotion")
    parser.add_argument("--valid", type=Path, default=Path("data/validation_clean.csv"), help="Path to cleaned VALIDATION split CSV with columns: text, emotion")
    parser.add_argument("--test",  type=Path, default=Path("data/test_clean.csv"), help="Path to cleaned TEST split CSV with columns: text, emotion")

    # Which models
    parser.add_argument("--models", nargs="+", default=["lr", "lsvm", "cnb"],
                        choices=["lr", "lsvm", "cnb"],
                        help="Subset of models to train. Default: all three.")

    # TF–IDF options
    parser.add_argument("--ngram-max", type=int, default=2, help="Use (1, ngram_max) word n-grams.")
    parser.add_argument("--max-features", type=int, default=50_000, help="Cap TF-IDF vocabulary size (most frequent features kept).")
    parser.add_argument("--min-df", type=int, default=2, help="Min documents a term must appear in to be kept (filters rare noise).")
    parser.add_argument("--disable-sublinear-tf", action="store_true", help="Disable sublinear TF scaling (default is enabled).")

    # Search options
    parser.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations per model.")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds in RandomizedSearchCV.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--search-class-weight", action="store_true", help="Include class_weight in LR/LinearSVC search (None vs 'balanced' instead of the default None).")

    # Output
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory to write serialized pipelines.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Directory to write JSON reports / top-terms.")
    parser.add_argument("--topk", type=int, default=20, help="Top-k n-grams per class to log for linear models.")

    args = parser.parse_args()

    rng = np.random.RandomState(args.random_seed)
    ensure_dirs(args.models_dir, args.reports_dir)

    # Load data
    train = pd.read_csv(args.train)
    valid = pd.read_csv(args.valid)
    test  = pd.read_csv(args.test)

    # Expect columns: text, label
    for df_name, df in [("train", train), ("validation", valid), ("test", test)]:
        if not {"text", "emotion"}.issubset(df.columns):
            raise ValueError(f"{df_name} missing required columns 'text' and 'emotion'.")

    X_train, y_train = train["text"].astype(str), train["emotion"].astype(str)
    X_val, y_val = valid["text"].astype(str), valid["emotion"].astype(str)
    X_test, y_test = test["text"].astype(str),  test["emotion"].astype(str)

    # Stable label order for metrics and confusion matrices
    labels_order = sorted(pd.concat([y_train, y_val, y_test]).unique().tolist())

    # Shared TF–IDF vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=not args.disable_sublinear_tf
    )

    # Model factory
    def make_pipeline(model_name: str) -> Pipeline:
        if model_name == "lr":
            clf = LogisticRegression(
                solver="saga",
                penalty="l2",
                max_iter=5000,
                tol=1e-3,
                n_jobs=-1
            )
        elif model_name == "lsvm":
            clf = LinearSVC(loss="hinge")
        elif model_name == "cnb":
            clf = ComplementNB()
        else:
            raise ValueError(model_name)
        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    # Train & evaluate each requested model on validation
    val_summaries = {}
    saved_model_paths = {}

    for model_name in args.models:
        pipe = make_pipeline(model_name)
        param_distributions = get_param_distributions(model_name, args.search_class_weight)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            scoring="f1_macro",
            cv=args.cv,
            n_jobs=-1,
            random_state=args.random_seed,
            verbose=1
        )

        t = timer()
        search.fit(X_train, y_train)
        train_time = t()

        best: Pipeline = search.best_estimator_ # Find the best pipeline for the specific model type   #TODO: Same note as for the val: tfidf - What does it mean when it's written like that? Why is it written like that?

        # Validation metrics + latency
        y_val_pred = best.predict(X_val)
        metrics_val = compute_metrics(y_val, y_val_pred, labels_order)
        latency_ms_val = latency_ms_per_sample(best, X_val)

        # Save pipeline immediately
        model_fname = {
            "lr": "shallow_lr.pkl",
            "lsvm": "shallow_lsvm.pkl",
            "cnb": "shallow_cnb.pkl"
        }[model_name]
        
        model_path = args.models_dir / model_fname
        joblib.dump(best, model_path)
        size_mb = model_file_size_mb(model_path)
        saved_model_paths[model_name] = str(model_path) 

        # Write per-model validation report
        report = {
            "model": {"name": {"lr":"logistic_regression","lsvm":"linear_svc","cnb":"complement_nb"}[model_name],
                      "best_params": search.best_params_},
            "vectorizer": {
                "type": "tfidf",
                "ngram_range": [1, args.ngram_max],
                "max_features": args.max_features,
                "min_df": args.min_df,
                "sublinear_tf": not args.disable_sublinear_tf,
                "strip_accents": "unicode"
            },
            "metrics": {
                "split": "validation",
                **metrics_val
            },
            "timing_sec": {"train": round(train_time, 4)},
            "latency_ms_per_sample": round(latency_ms_val, 4),
            "model_size_mb": size_mb,
            "random_seed": args.random_seed
        }

        out_path = args.reports_dir / {
            "lr": "shallow_lr_val.json",
            "lsvm":"shallow_lsvm_val.json",
            "cnb":"shallow_cnb_val.json"
        }[model_name]
        save_json(out_path, report)

        # Optional: write top-k terms for linear models
        if model_name in {"lr", "lsvm"} and args.topk > 0:
            txt = extract_top_terms_txt(best, labels_order, k=args.topk)
            if txt:
                top_path = args.reports_dir / {
                    "lr": "shallow_top_features_lr.txt",
                    "lsvm": "shallow_top_features_lsvm.txt"
                }[model_name]
                with top_path.open("w", encoding="utf-8") as f:
                    f.write(txt)

        # Keep summary for winner selection
        val_summaries[model_name] = {
            "macro_f1": report["metrics"]["macro_f1"],
            "accuracy": report["metrics"]["accuracy"],
            "model_path": str(model_path)
        }

    # Select winner: highest macro-F1, tie-break by accuracy
    def sort_key(item):
        m = item[1]
        return (m["macro_f1"], m["accuracy"])

    winner_name, _ = sorted(val_summaries.items(), key=sort_key, reverse=True)[0]

    # Retrain winner on train + validation, evaluate on test
    winner_pipe = make_pipeline(winner_name)
    # Use the best params found on validation search for the winner
    # Reload from saved best pipeline to preserve exact params
    winner_best_path = {
        "lr": args.models_dir / "shallow_lr.pkl",
        "lsvm": args.models_dir / "shallow_lsvm.pkl",
        "cnb": args.models_dir / "shallow_cnb.pkl"
    }[winner_name]
    winner_best: Pipeline = joblib.load(winner_best_path)

    # Refit on train + valid to use all available labeled data before testing
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)  # TODO: Maybe change this shortened name to somtehing that goes better with python conventions? Maybe do this for all the variables that need the same treatment?
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    t = timer()
    winner_best.fit(X_train_val, y_train_val)
    train_time = t()

    y_test_pred = winner_best.predict(X_test)
    metrics_test = compute_metrics(y_test, y_test_pred, labels_order)
    latency_ms_test = latency_ms_per_sample(winner_best, X_test)
    size_mb_test = model_file_size_mb(winner_best_path)

    # Save a convenience copy for the winner
    joblib.dump(winner_best, args.models_dir / "shallow_winner.pkl")

    winner_params = {}
    # Try to pull params from the stored best estimator
    clf = winner_best.named_steps["clf"]
    # Keep only the params relevant to the model to keep the JSON tidy
    if winner_name == "lr":
        keep = ["C", "penalty", "solver", "multi_class", "class_weight", "max_iter"]
        winner_params = {k: getattr(clf, k, None) for k in keep}
    elif winner_name == "lsvm":
        keep = ["C", "loss", "class_weight"]
        winner_params = {k: getattr(clf, k, None) for k in keep}
    else:  # cnb
        keep = ["alpha"]
        winner_params = {k: getattr(clf, k, None) for k in keep}

    # Write winner test report
    winner_report = {
        "winner_model": {"lr":"logistic_regression","lsvm":"linear_svc","cnb":"complement_nb"}[winner_name],
        "winner_params": winner_params,
        "metrics": {"split": "test", **metrics_test},
        "timing_sec": {"train": round(train_time, 4)},
        "latency_ms_per_sample": round(latency_ms_test, 4),
        "model_size_mb": size_mb_test,
    }
    save_json(args.reports_dir / "shallow_winner_test.json", winner_report)

    print(f"Winner: {winner_report['winner_model']}")
    print(f"Validation macro-F1 (per model): " +
          ", ".join([f"{k}:{val_summaries[k]['macro_f1']:.3f}" for k in args.models]))
    print("Artifacts written to:")
    print(f"  models/: {', '.join(sorted(os.listdir(args.models_dir)))}")
    print(f"  reports/: {', '.join(sorted(os.listdir(args.reports_dir)))}")
    
        
    # TODO: Make sure to add the explanation that because "macro F1" ensures in the actual comparison that the winning model makes good result for each class individually and not just in total, thus deals with the case of bias because of imalanced dataset that is not taken care of
    # Do that somewhere in the scripts, not too long, just when used for example
    
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------ QUESTIONS FOR NEXT SESSION ------------------------------------------------------------------------------------------
# """    
# * I got this warning: "C:\Users\Inbal\anaconda3\envs\env_emotion_detection\lib\site-packages\sklearn\linear_model\_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge"

# * I also got this warning: "C:\Users\Inbal\anaconda3\envs\env_emotion_detection\lib\site-packages\sklearn\svm\_base.py:1250: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations."

# * I'm not sure I understood: Does the number for --ngram-max means all the number of grams from 1 to the chosen number include? Meaning If I chose --ngram-max=5 it will go through all 1, 2, 3, 4, 5 grams?

# * Why in --max_features :     parser.add_argument("--max-features", type=int, default=50_000, help="Cap TF-IDF vocabulary size (most frequent features kept).")

# * default is written like this: 50_000 and not like this 50000? Does it read it like that? I don't understand.

# * I don't understand what min-df does or referes to

# * I don't understand what --no-sublinear-tf means if disabled or enabled, what does it affect?

# * What is args.model_dir? I don't seem to find it
# """


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------ TODO's ------------------------------------------------------------------------------------------
# """
# 1. Answer the questions
# 2. Fix the warnings
# 3. In the notebook part - Compare between the models based on several parameters, including the complexities
#    Keep in mind that the current winner in this script is selected based on the macro-F1 score, with accuracy as a tie-braker accuracy.