"""
Iris flower classification.

Compares Logistic Regression, KNN, and a Decision Tree on the classic
Iris dataset. Saves plots, metrics, and the best fitted pipeline.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target.map(lambda i: iris.target_names[i])
    return X, y, list(iris.target_names)


def plot_eda(X: pd.DataFrame, y: pd.Series) -> None:
    df = X.copy()
    df["species"] = y

    pair = sns.pairplot(
        df,
        hue="species",
        corner=True,
        diag_kind="kde",
        plot_kws={"s": 40, "alpha": 0.8},
    )
    pair.fig.suptitle("Iris feature relationships by species", y=1.02, fontsize=16)
    pair.savefig(OUTPUT_DIR / "01_pairplot.png", dpi=160, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = X.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_correlation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=500),
                ),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("clf", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": cv_scores.mean(),
                "cv_accuracy_std": cv_scores.std(),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision_macro": precision_score(
                    y_test, y_pred, average="macro"
                ),
                "test_recall_macro": recall_score(y_test, y_pred, average="macro"),
                "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
            }
        )
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))

    return pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)


def plot_confusion(model: Pipeline, X_test, y_test, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=labels,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Best model — confusion matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    X, y, labels = load_data()
    print("Dataset shape:", X.shape)
    print("Class balance:\n", y.value_counts().to_string())

    plot_eda(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best_name = results.iloc[0]["model"]
    best_model = models[best_name]
    # already fitted inside evaluate_models
    plot_confusion(best_model, X_test, y_test, labels)
    joblib.dump(best_model, OUTPUT_DIR / "best_iris_model.joblib")

    report = f"""Iris Flower Classification — Results
=====================================
Best model: {best_name}
Test accuracy: {results.iloc[0]['test_accuracy']:.3f}
Macro F1: {results.iloc[0]['test_f1_macro']:.3f}
5-fold CV accuracy: {results.iloc[0]['cv_accuracy_mean']:.3f} ± {results.iloc[0]['cv_accuracy_std']:.3f}

Notes
-----
- Features: sepal length/width, petal length/width (cm)
- Target: setosa, versicolor, virginica
- Train/test split: 80/20, stratified
- Logistic Regression and KNN use StandardScaler
- Petal length and petal width separate the species most clearly
"""
    (OUTPUT_DIR / "results_summary.txt").write_text(report)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
