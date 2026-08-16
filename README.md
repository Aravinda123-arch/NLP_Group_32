# Fake News Detection — Logistic Regression & LSTM

This branch contains the implementation and evaluation of **Logistic Regression** and **Long Short-Term Memory (LSTM)** models for the Fake News Detection project.

The models use Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques to classify news articles as **Fake** or **Real**.

---

## Project Information

**Repository:** [`NLP_Group_32`](https://github.com/Aravinda123-arch/NLP_Group_32)

**Branch:**

```text
feature/CIT-24-01-0427-Logistic-Regression+LSTM
```

**Student ID:** `CIT-24-01-0427`

**Models:**

* Logistic Regression
* LSTM

---

## Problem Statement

The rapid spread of fake news and misinformation through online platforms makes it difficult for users to determine whether news articles are reliable.

The objective of this project is to develop an automated Fake News Detection system that can classify news articles as **Fake** or **Real** using their textual content.

This branch focuses on implementing and evaluating **Logistic Regression** and **LSTM** models for this classification task.

---

## Dataset Information

The project uses two CSV files:

```text
data/
├── Fake.csv
└── True.csv
```

### Fake.csv

Contains news articles classified as **Fake**.

### True.csv

Contains news articles classified as **True**.

The datasets are used as the input for the NLP preprocessing and model training pipelines.

---

## Repository Structure

```text
NLP_Group_32/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/
│   ├── LogisticRegression.ipynb
│   └── LSTM.ipynb
│
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── preprocess_data.py
│   ├── train_logistic_regression.py
│   ├── train_lstm.py
│   ├── logistic_regression_external_test.py
│   └── lstm_external_test.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lstm_model.keras
│   ├── lstm_tokenizer.pickle
│   └── confusion_matrix.png
│
├── reports/
│   ├── logistic_regression_polyglot_external_metrics.csv
│   ├── logistic_regression_polyglot_external_predictions.csv
│   ├── logistic_regression_polyglot_confusion_matrix.csv
│   ├── lstm_polyglot_external_metrics.csv
│   ├── lstm_polyglot_external_predictions.csv
│   ├── lstm_polyglot_confusion_matrix.csv
│   └── lstm_polyglot_classification_report.csv
│
├── screenshots/
├── videos/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## NLP Processing Pipeline

The general workflow used for the models is:

```text
Raw News Dataset
       ↓
Data Loading (src/load_data.py)
       ↓
Data Cleaning (src/clean_data.py)
       ↓
Text Preprocessing (src/preprocess_data.py)
       ↓
Feature Extraction / Tokenization
       ↓
Train-Test Split
       ↓
Model Training (src/train_logistic_regression.py, src/train_lstm.py)
       ↓
Prediction
       ↓
Model Evaluation
       ↓
External Testing (src/logistic_regression_external_test.py, src/lstm_external_test.py)
```

---

## Source Code (`src/`)

The `src/` directory contains modular Python scripts that implement the full pipeline:

| Script                                | Description                                                        |
| ------------------------------------- | ------------------------------------------------------------------ |
| `load_data.py`                        | Loads the Fake.csv and True.csv datasets                           |
| `clean_data.py`                       | Cleans raw text data (removes noise, special characters)           |
| `preprocess_data.py`                  | NLP preprocessing (tokenization, stopword removal, lemmatization)  |
| `train_logistic_regression.py`        | Trains the Logistic Regression model with TF-IDF features          |
| `train_lstm.py`                       | Trains the LSTM deep learning model with tokenized sequences       |
| `logistic_regression_external_test.py`| Evaluates Logistic Regression on the external Polyglot dataset     |
| `lstm_external_test.py`              | Evaluates LSTM on the external Polyglot dataset                    |

---

## Models Implemented

### 1. Logistic Regression

Logistic Regression is a supervised machine learning classification algorithm used to classify news articles based on extracted textual features.

**Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)

**Saved Model Artifacts:**

```text
models/logistic_regression.pkl
models/tfidf_vectorizer.pkl
```

The implementation is available in:

```text
notebooks/LogisticRegression.ipynb
src/train_logistic_regression.py
```

### 2. LSTM

Long Short-Term Memory (LSTM) is a recurrent neural network architecture designed to learn sequential patterns and dependencies within textual data.

**Feature Extraction:** Keras Tokenizer with padding sequences

**Saved Model Artifacts:**

```text
models/lstm_model.keras
models/lstm_tokenizer.pickle
```

The implementation is available in:

```text
notebooks/LSTM.ipynb
src/train_lstm.py
```

---

## Model Summary

| Model               | Type             | Notebook                   | Training Script                  |
| ------------------- | ---------------- | -------------------------- | -------------------------------- |
| Logistic Regression | Machine Learning | `LogisticRegression.ipynb` | `train_logistic_regression.py`   |
| LSTM                | Deep Learning    | `LSTM.ipynb`               | `train_lstm.py`                  |

---

## Streamlit Web Application

An interactive **Streamlit** web application (`app.py`) is included for demonstration purposes, providing a portal to test the fake news detection models.

### Features

* Model selection across all team members' models
* Text input for news article analysis
* Real-time prediction with confidence scores
* Decision boundary feature analysis visualization
* Premium glassmorphic dark-mode UI

### How to Run the Web App

```bash
streamlit run app.py
```

---

## External Test Results

The models were evaluated using an **external test dataset containing 6,642 records**.

The external evaluation includes accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, and Matthews Correlation Coefficient (MCC).

## Logistic Regression — External Test

| Metric            |      Result |
| ----------------- | ----------: |
| External Records  |       6,642 |
| External Accuracy |  **49.35%** |
| Balanced Accuracy |  **49.63%** |
| Macro Precision   |  **0.4964** |
| Macro Recall      |  **0.4963** |
| Macro F1-Score    |  **0.4900** |
| Weighted F1-Score |  **0.4983** |
| ROC-AUC           |  **0.4913** |
| MCC               | **-0.0073** |

### Classification Report

| Class                |  Precision |     Recall |   F1-Score |   Support |
| -------------------- | ---------: | ---------: | ---------: | --------: |
| Fake                 |     0.3987 |     0.5103 |     0.4476 |     2,671 |
| Real                 |     0.5942 |     0.4822 |     0.5324 |     3,971 |
| **Accuracy**         |            |            | **0.4935** | **6,642** |
| **Macro Average**    | **0.4964** | **0.4963** | **0.4900** | **6,642** |
| **Weighted Average** | **0.5155** | **0.4935** | **0.4983** | **6,642** |

### Confusion Matrix

```text
[[1363 1308]
 [2056 1915]]
```

---

## LSTM — External Test

| Metric            |      Result |
| ----------------- | ----------: |
| External Records  |       6,642 |
| External Accuracy |  **41.81%** |
| Balanced Accuracy |  **47.50%** |
| Macro Precision   |  **0.4631** |
| Macro Recall      |  **0.4750** |
| Macro F1-Score    |  **0.3944** |
| Weighted F1-Score |  **0.3709** |
| ROC-AUC           |  **0.4480** |
| MCC               | **-0.0607** |

### Classification Report

| Class             |  Precision |     Recall |   F1-Score |   Support |
| ----------------- | ---------: | ---------: | ---------: | --------: |
| Fake              |     0.3871 |     0.7660 |     0.5143 |     2,671 |
| Real              |     0.5391 |     0.1841 |     0.2745 |     3,971 |
| **Accuracy**      |            |            | **0.4181** | **6,642** |
| **Macro Average** | **0.4631** | **0.4750** | **0.3944** | **6,642** |

---

## Overall Results Comparison

| Model                   |   Accuracy | Balanced Accuracy |   Macro F1 | Weighted F1 |    ROC-AUC |         MCC |
| ----------------------- | ---------: | ----------------: | ---------: | ----------: | ---------: | ----------: |
| **Logistic Regression** | **49.35%** |        **49.63%** | **0.4900** |  **0.4983** | **0.4913** | **-0.0073** |
| **LSTM**                | **41.81%** |        **47.50%** | **0.3944** |  **0.3709** | **0.4480** | **-0.0607** |

### Result Observation

Based on the external test results, **Logistic Regression performed better than LSTM** on this evaluation dataset.

Logistic Regression achieved:

* Higher accuracy (**49.35% vs. 41.81%**)
* Higher macro F1-score (**0.4900 vs. 0.3944**)
* Higher weighted F1-score (**0.4983 vs. 0.3709**)
* Higher ROC-AUC (**0.4913 vs. 0.4480**)
* Higher MCC (**-0.0073 vs. -0.0607**)

However, both models achieved performance close to or below 50% accuracy on the external test set, indicating that the models did not generalize strongly to this external dataset.

---

## Generated Reports

The external test scripts produce detailed evaluation reports saved in the `reports/` directory:

| Report File                                            | Description                                           |
| ------------------------------------------------------ | ----------------------------------------------------- |
| `logistic_regression_polyglot_external_metrics.csv`    | Evaluation metrics for Logistic Regression             |
| `logistic_regression_polyglot_external_predictions.csv`| Full predictions on the external dataset (LR)          |
| `logistic_regression_polyglot_confusion_matrix.csv`    | Confusion matrix for Logistic Regression               |
| `lstm_polyglot_external_metrics.csv`                   | Evaluation metrics for LSTM                            |
| `lstm_polyglot_external_predictions.csv`               | Full predictions on the external dataset (LSTM)        |
| `lstm_polyglot_confusion_matrix.csv`                   | Confusion matrix for LSTM                              |
| `lstm_polyglot_classification_report.csv`              | Classification report for LSTM                         |

---

## Evaluation Metrics

The following metrics are used to evaluate the models:

* **Accuracy** — Overall proportion of correctly classified records.
* **Balanced Accuracy** — Average recall across the classes.
* **Precision** — Proportion of predicted samples that belong to the corresponding class.
* **Recall** — Proportion of actual samples correctly identified.
* **F1-Score** — Harmonic mean of precision and recall.
* **ROC-AUC** — Measures the model's ability to distinguish between the classes.
* **MCC** — Measures the quality of binary classifications using all four confusion-matrix categories.

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* NLTK
* Streamlit
* Matplotlib
* Seaborn
* Joblib
* Natural Language Processing (NLP)
* Git
* GitHub

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Aravinda123-arch/NLP_Group_32.git
```

Navigate to the repository:

```bash
cd NLP_Group_32
```

### 2. Switch to This Branch

```bash
git checkout feature/CIT-24-01-0427-Logistic-Regression+LSTM
```

Or:

```bash
git switch feature/CIT-24-01-0427-Logistic-Regression+LSTM
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Check Dataset

Make sure the following files are available:

```text
data/Fake.csv
data/True.csv
```

---

## How to Run

### Jupyter Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open either:

```text
notebooks/LogisticRegression.ipynb
```

or:

```text
notebooks/LSTM.ipynb
```

Run the notebook cells sequentially to perform:

1. Dataset loading
2. Text preprocessing
3. Feature extraction / tokenization
4. Model training
5. Prediction
6. Model evaluation

### Python Scripts

Run training scripts directly:

```bash
python src/train_logistic_regression.py
python src/train_lstm.py
```

Run external test evaluation:

```bash
python src/logistic_regression_external_test.py
python src/lstm_external_test.py
```

### Streamlit Web App

```bash
streamlit run app.py
```

---

## Git Commit Guidelines

Meaningful commit messages should be used to clearly describe completed work.

### Examples

```text
Added text preprocessing pipeline
Implemented TF-IDF feature extraction
Trained Logistic Regression model
Implemented LSTM model
Added external test evaluation
Added evaluation metrics
Fixed preprocessing issue
Updated model results
```

The lecturer's guidelines recommend descriptive commit messages and identify vague messages such as `update`, `work`, `final`, and `done` as poor examples.

---

## Git Requirements

The project follows the Git requirements provided by the lecturer:

* Every student should have visible commits.
* Meaningful repository activity should be maintained.
* Commit timestamps may be checked.
* Mass uploads near the deadline should be avoided.
* Branching and merging practices will be evaluated.

---

## Contributor

**Student ID:** `CIT-24-01-0427`

**Contribution:**

* Logistic Regression model (training + external evaluation)
* LSTM model (training + external evaluation)
* NLP preprocessing pipeline
* External test evaluation scripts
* Streamlit web application
* Generated evaluation reports

**Branch:**

```text
feature/CIT-24-01-0427-Logistic-Regression+LSTM
```

---

## License

This project is developed for academic and educational purposes.
