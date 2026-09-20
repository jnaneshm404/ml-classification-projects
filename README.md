# Supervised Learning Projects

Two small classification projects in Python and scikit-learn.

I built these to practice the full loop: load data, explore it, train more than one baseline, pick metrics that match the problem, and save plots I can explain.

Not a company internship. Just the work.

## Projects

### 1. Iris flower classification
`01_iris_classification/`

Predict species (`setosa`, `versicolor`, `virginica`) from sepal and petal measurements.

- Dataset: [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) (150 rows, 4 features, balanced)
- Models: Logistic Regression, KNN (k=5), Decision Tree
- Split: stratified 80/20
- Extra: 5-fold stratified CV on the training set, StandardScaler inside a Pipeline

**Results**

| Model | CV accuracy | Test accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.958 ± 0.026 | 0.933 | 0.933 |
| Decision Tree | 0.942 ± 0.020 | 0.933 | 0.933 |
| KNN | 0.958 ± 0.026 | 0.933 | 0.933 |

Petal length and petal width do most of the work. Setosa is linearly separable. Versicolor and virginica overlap a little, which is where the remaining errors come from.

### 2. SMS spam detector
`02_spam_detector/`

Classify text messages as `spam` or `ham`.

- Dataset: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,572 messages, ~13% spam)
- Features: TF-IDF unigrams + bigrams, English stopwords
- Models: Multinomial Naive Bayes vs class-weighted Logistic Regression
- Metrics: spam-class precision / recall / F1 (overall accuracy is misleading here)

**Results**

| Model | Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.976 | 0.896 | 0.926 | 0.911 |
| Naive Bayes | 0.970 | 1.000 | 0.779 | 0.875 |

Naive Bayes almost never calls ham “spam,” but it misses more actual spam. Logistic Regression is the better default if the cost of missing spam is higher.

Tokens with the largest positive weights for spam included `txt`, `claim`, `free`, `prize`, `150p`, and `www`.

## Run it

```bash
python -m pip install -r requirements.txt
python 01_iris_classification/iris_classification.py
python 02_spam_detector/spam_detector.py
```

The spam script downloads the public dataset on first run and caches it under `02_spam_detector/data/`.

Plots, CSVs, and saved models go in each project’s `outputs/` folder.

## What this is good for

- Showing you can write a clean Pipeline
- Showing you compare baselines instead of reporting one score
- Showing you know accuracy is the wrong headline metric on imbalanced text

## What this is not

- Not production ML
- Not deep learning
- Not work done inside a company

If you are reviewing this repo, the files to read first are the two `.py` scripts and the plots in `outputs/`.
