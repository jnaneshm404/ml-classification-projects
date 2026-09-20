"""
SMS spam vs ham classifier.

Downloads the public SMS Spam Collection, vectorizes messages with TF-IDF,
and compares Multinomial Naive Bayes with class-weighted Logistic Regression.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SMS_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)
LOCAL_TSV = DATA_DIR / "sms.tsv"

sns.set_theme(style="whitegrid", context="talk")


def load_sms() -> pd.DataFrame:
    if not LOCAL_TSV.exists():
        print("Downloading SMS Spam Collection...")
        with urllib.request.urlopen(SMS_URL, timeout=30) as resp:
            LOCAL_TSV.write_bytes(resp.read())
    df = pd.read_csv(LOCAL_TSV, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].str.lower().str.strip()
    df["message"] = df["message"].astype(str)
    df = df.dropna(subset=["label", "message"])
    return df


def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["label"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette=["#2a9d8f", "#e76f51"])
    axes[0].set_title("Class count")
    axes[0].set_ylabel("messages")

    df = df.copy()
    df["n_chars"] = df["message"].str.len()
    sns.kdeplot(
        data=df,
        x="n_chars",
        hue="label",
        fill=True,
        common_norm=False,
        ax=axes[1],
        clip=(0, 300),
    )
    axes[1].set_title("Message length by class")
    axes[1].set_xlabel("characters")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_eda.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_models() -> dict[str, Pipeline]:
    tfidf = dict(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=15000,
    )
    return {
        "Naive Bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf)),
                ("clf", MultinomialNB()),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=400,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        ),
    }


def evaluate(
    models: dict[str, Pipeline],
    X_train,
    X_test,
    y_train,
    y_test,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_spam": precision_score(
                    y_test, y_pred, pos_label="spam"
                ),
                "recall_spam": recall_score(y_test, y_pred, pos_label="spam"),
                "f1_spam": f1_score(y_test, y_pred, pos_label="spam"),
            }
        )
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, digits=3))
    return pd.DataFrame(rows).sort_values("f1_spam", ascending=False)


def plot_confusion(model: Pipeline, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["ham", "spam"],
        cmap="Oranges",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Best model — confusion matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def show_top_tokens(model: Pipeline, n: int = 15) -> pd.DataFrame:
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = vectorizer.get_feature_names_out()
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
        top_spam = pd.DataFrame(
            {"token": names, "weight": weights}
        ).sort_values("weight", ascending=False).head(n)
        top_ham = pd.DataFrame(
            {"token": names, "weight": weights}
        ).sort_values("weight", ascending=True).head(n)
        top_spam.to_csv(OUTPUT_DIR / "top_spam_tokens.csv", index=False)
        top_ham.to_csv(OUTPUT_DIR / "top_ham_tokens.csv", index=False)
        print("\nTokens most associated with spam:")
        print(top_spam.to_string(index=False))
        return top_spam
    return pd.DataFrame()


def demo_predict(model: Pipeline) -> None:
    samples = [
        "Congratulations! You won a FREE lottery prize. Call now to claim your cash.",
        "Hey, are we still meeting at 6 for dinner?",
        "URGENT: Your bank account will be closed. Verify at http://bit.ly/not-real",
        "Can you send the assignment notes when you get home?",
    ]
    preds = model.predict(samples)
    lines = ["Sample predictions", "------------------"]
    for text, label in zip(samples, preds):
        lines.append(f"[{label.upper():4}] {text}")
    (OUTPUT_DIR / "sample_predictions.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


def main() -> None:
    df = load_sms()
    print("Rows:", len(df))
    print(df["label"].value_counts())

    plot_eda(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    models = build_models()
    results = evaluate(models, X_train, X_test, y_train, y_test)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best_name = results.iloc[0]["model"]
    best_model = models[best_name]
    plot_confusion(best_model, X_test, y_test)
    show_top_tokens(best_model)
    demo_predict(best_model)
    joblib.dump(best_model, OUTPUT_DIR / "best_spam_model.joblib")

    summary = f"""Spam Mail Detector — Results
=============================
Dataset: SMS Spam Collection
Rows: {len(df)}
Best model: {best_name}
Accuracy: {results.iloc[0]['accuracy']:.3f}
Spam precision: {results.iloc[0]['precision_spam']:.3f}
Spam recall: {results.iloc[0]['recall_spam']:.3f}
Spam F1: {results.iloc[0]['f1_spam']:.3f}

Pipeline
--------
1. Lowercase + English stopword removal
2. TF-IDF unigrams + bigrams
3. Naive Bayes vs class-weighted Logistic Regression
4. Report spam-class metrics (more important than overall accuracy)
"""
    (OUTPUT_DIR / "results_summary.txt").write_text(summary)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
