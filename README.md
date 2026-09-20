# Supervised Learning Projects

Classification projects in Python using scikit-learn.

## 1. Iris flower classification

Predicts iris species (`setosa`, `versicolor`, `virginica`) from sepal and petal measurements.

- Dataset: [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) — 150 samples, 4 features, 3 balanced classes
- Models: Logistic Regression, K-Nearest Neighbors (`k=5`), Decision Tree
- Evaluation: stratified 80/20 split, 5-fold cross-validation, precision / recall / F1, confusion matrix
- Preprocessing: `StandardScaler` inside a `Pipeline` for Logistic Regression and KNN

| Model | CV accuracy | Test accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.958 ± 0.026 | 0.933 | 0.933 |
| Decision Tree | 0.942 ± 0.020 | 0.933 | 0.933 |
| KNN | 0.958 ± 0.026 | 0.933 | 0.933 |

Petal length and petal width separate the classes most clearly. Setosa is linearly separable. Most leftover errors are between versicolor and virginica.

Code: `01_iris_classification/iris_classification.py`  
Plots and metrics: `01_iris_classification/outputs/`

## 2. SMS spam detector

Classifies text messages as spam or ham.

- Dataset: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 messages, about 13% spam
- Features: TF-IDF unigrams and bigrams, English stopwords
- Models: Multinomial Naive Bayes and class-weighted Logistic Regression
- Evaluation: accuracy plus spam-class precision, recall, and F1

| Model | Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.976 | 0.896 | 0.926 | 0.911 |
| Naive Bayes | 0.970 | 1.000 | 0.779 | 0.875 |

Accuracy is high for both models because most messages are ham. Spam F1 is the more useful score. Naive Bayes is conservative (no false spam in this test set, lower recall). Logistic Regression catches more spam.

Tokens with the strongest spam weights included `txt`, `claim`, `free`, `prize`, `150p`, and `www`.

Code: `02_spam_detector/spam_detector.py`  
Plots and metrics: `02_spam_detector/outputs/`

## Setup

```bash
python -m pip install -r requirements.txt
python 01_iris_classification/iris_classification.py
python 02_spam_detector/spam_detector.py
```

The spam script downloads the public dataset on first run and saves it under 02_spam_detector/data/.
