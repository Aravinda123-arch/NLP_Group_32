# Fake News Detection — SVM & GRU

This branch contains the implementation and evaluation of **Support Vector Machine (SVM)** and **Gated Recurrent Unit (GRU)** models for the Fake News Detection project.

The models use Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques to classify news articles as **Fake** or **Real**.

---

## Project Information

**Repository:** `NLP_Group_32`

**Branch:**

```text
feature/CIT-24-01-0363-SVM+CNN
```

**Student ID:** `CIT-24-01-0363`

**Models evaluated:**

* Support Vector Machine (SVM)
* Gated Recurrent Unit (GRU)

> **Note:** The branch name contains `SVM+CNN`, while the external test results provided for this update are for **SVM and GRU**. The results section therefore reports the actual SVM and GRU evaluation results without changing or assuming the CNN results.

---

## Problem Statement

The rapid spread of fake news and misinformation through online platforms makes it difficult for users to determine whether news articles are reliable.

The objective of this project is to develop an automated Fake News Detection system that can classify news articles as **Fake** or **Real** using their textual content.

This branch focuses on implementing and evaluating **SVM** and **GRU** models for this classification task.

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

Contains news articles classified as **Real**.

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
│   ├── SVM.ipynb
│   └── GRU.ipynb
│
├── src/
├── models/
├── reports/
├── screenshots/
├── videos/
├── requirements.txt
├── README.md
└── .gitignore
```

The repository structure follows the structure specified in the lecturer's Git Guidelines.

---

## NLP Processing Pipeline

The general workflow used for the models is:

```text
Raw News Dataset
       ↓
Data Loading
       ↓
Text Preprocessing
       ↓
Feature Extraction / Tokenization
       ↓
Train-Test Split
       ↓
Model Training
       ↓
External Test Prediction
       ↓
Model Evaluation
```

The preprocessing and feature representation may differ depending on the model.

---

# Models Implemented

## 1. Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised machine learning algorithm used to classify news articles by finding a decision boundary between the Fake and Real classes.

The SVM model was evaluated using the Polyglot external test dataset.

### SVM External Test

```text
External records: 6642
```

The model generated predictions using the SVM decision function for ROC-AUC calculation.

---

## 2. Gated Recurrent Unit (GRU)

Gated Recurrent Unit (GRU) is a recurrent neural network architecture designed to learn sequential patterns and dependencies within textual data.

The GRU model was evaluated using the Polyglot external test dataset.

### GRU External Test

```text
External records: 6760
Raw prediction shape: (6760, 1)
```

---

# Model Summary

| Model | Type             | Evaluation Dataset     |
| ----- | ---------------- | ---------------------- |
| SVM   | Machine Learning | Polyglot External Test |
| GRU   | Deep Learning    | Polyglot External Test |

---

# Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Natural Language Processing (NLP)
* Matplotlib
* Seaborn
* Git
* GitHub

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/Aravinda123-arch/NLP_Group_32.git
```

Navigate to the project directory:

```bash
cd NLP_Group_32
```

## 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 3. Check Dataset

Make sure the dataset files are available inside the `data/` directory:

```text
data/
├── Fake.csv
└── True.csv
```

---

# Switch to This Branch

Use:

```bash
git checkout feature/CIT-24-01-0363-SVM+CNN
```

Or:

```bash
git switch feature/CIT-24-01-0363-SVM+CNN
```

The lecturer's guidelines require each team member to work using a separate branch following the `feature/memberID-model` convention.

---

# How to Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open the appropriate notebook from the `notebooks/` directory.

For SVM:

```text
notebooks/SVM.ipynb
```

For GRU:

```text
notebooks/GRU.ipynb
```

Run the notebook cells sequentially to perform:

1. Dataset loading
2. Text preprocessing
3. Feature extraction / tokenization
4. Model training
5. Prediction
6. External test evaluation

---

# External Test Results

The models were evaluated using **Polyglot external test data**.

The evaluation includes:

* Accuracy
* Balanced Accuracy
* Macro Precision
* Macro Recall
* Macro F1-Score
* Weighted F1-Score
* ROC-AUC
* Matthews Correlation Coefficient (MCC)

---

## SVM — Polyglot External Test

The SVM external test was performed on **6,642 external records**.

| Metric            |     Result |
| ----------------- | ---------: |
| External Records  |  **6,642** |
| External Accuracy | **50.60%** |
| Balanced Accuracy | **0.5280** |
| Macro Precision   | **0.5282** |
| Macro Recall      | **0.5280** |
| Macro F1-Score    | **0.5060** |
| Weighted F1-Score | **0.5051** |
| ROC-AUC           | **0.5233** |
| MCC               | **0.0562** |

### ROC-AUC

The ROC-AUC score was calculated using the model's:

```text
decision_function
```

---

## GRU — Polyglot External Test

The GRU external test was performed on **6,760 external records**.

The raw prediction output shape was:

```text
(6760, 1)
```

| Metric            |     Result |
| ----------------- | ---------: |
| External Records  |  **6,760** |
| External Accuracy | **50.03%** |
| Balanced Accuracy | **0.5060** |
| Macro Precision   | **0.5058** |
| Macro Recall      | **0.5060** |
| Macro F1-Score    | **0.4975** |
| Weighted F1-Score | **0.5051** |
| ROC-AUC           | **0.5326** |
| MCC               | **0.0118** |

---

# Overall Results Comparison

| Model   | External Records |   Accuracy | Balanced Accuracy |   Macro F1 | Weighted F1 |    ROC-AUC |        MCC |
| ------- | ---------------: | ---------: | ----------------: | ---------: | ----------: | ---------: | ---------: |
| **SVM** |            6,642 | **50.60%** |        **0.5280** | **0.5060** |  **0.5051** |     0.5233 | **0.0562** |
| **GRU** |            6,760 |     50.03% |            0.5060 |     0.4975 |  **0.5051** | **0.5326** |     0.0118 |

---

# Results Observation

Based on the provided Polyglot external test results:

* **SVM achieved the higher external accuracy**, with **50.60%** compared with **50.03%** for GRU.
* **SVM achieved higher balanced accuracy**, with **0.5280** compared with **0.5060**.
* **SVM achieved higher macro precision**, with **0.5282** compared with **0.5058**.
* **SVM achieved higher macro recall**, with **0.5280** compared with **0.5060**.
* **SVM achieved a higher macro F1-score**, with **0.5060** compared with **0.4975**.
* Both models achieved the same **weighted F1-score of 0.5051**.
* **GRU achieved the higher ROC-AUC**, with **0.5326** compared with **0.5233** for SVM.
* **SVM achieved the higher MCC**, with **0.0562** compared with **0.0118** for GRU.

Overall, the SVM model performed slightly better than GRU across most of the reported classification metrics, while GRU achieved the higher ROC-AUC.

The results are close to 50% accuracy, indicating that both models have limited classification performance on the provided external test datasets.

> **Important:** The SVM and GRU external test sets contain different numbers of records (6,642 and 6,760 respectively). Therefore, the results should be interpreted as the reported performance for each model's respective external test set.

---

# Classification Report

The detailed **classification report values were not included in the supplied results for this update**.

Therefore, the README does not include fabricated precision, recall, F1-score, or support values for individual Fake/Real classes.

Once the complete classification reports are available, they can be added here.

---

# Evaluation Metrics

### Accuracy

Measures the proportion of correctly classified samples among all evaluated samples.

### Balanced Accuracy

Measures the average recall across the classes and is useful when class distributions are not perfectly balanced.

### Precision

Measures how many of the samples predicted as a particular class were actually members of that class.

### Recall

Measures how many of the actual samples belonging to a class were correctly identified.

### F1-Score

Combines precision and recall into a single metric.

### ROC-AUC

Measures the ability of the model to distinguish between the two classes.

### Matthews Correlation Coefficient (MCC)

Provides a correlation-based measure of binary classification performance using the complete confusion matrix.

---

# Git Commit Guidelines

Meaningful commit messages should clearly describe the work completed.

### Recommended Commit Messages

```text
Added text preprocessing pipeline
Implemented TF-IDF feature extraction
Trained SVM model
Implemented GRU model
Added Polyglot external test
Added external evaluation metrics
Fixed preprocessing issue
Updated model evaluation results
```

The lecturer's guidelines recommend descriptive commit messages such as `Added text preprocessing pipeline`, `Implemented TF-IDF feature extraction`, and `Trained LSTM model`.

Avoid vague commit messages such as:

```text
update
work
final
done
```

These are specifically identified as poor examples in the lecturer's guidelines.

---

# Git Requirements

The project follows the Git practices specified in the lecturer's guidelines:

* Every student should have visible commits.
* Meaningful repository activity should be maintained.
* Commit timestamps may be checked.
* Mass uploads near the deadline should be avoided.
* Branching and merging practices will be evaluated.

---

# Contributor

**Student ID:** `CIT-24-01-0363`

**Contribution:**

* Support Vector Machine (SVM)
* Gated Recurrent Unit (GRU)

**Branch:**

```text
feature/CIT-24-01-0363-SVM+CNN
```

---

# License

This project is developed for **academic and educational purposes**.