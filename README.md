# 📰 Fake News Detection System

An end-to-end Natural Language Processing (NLP) framework and interactive web application developed to classify news articles as **Factual (Real)** or **Fabricated (Fake)** using classical Machine Learning, Deep Learning (RNNs), and Transformer models (BERT).

---

## 👥 Group Members (NLP Group 03)

| Member | Student Registration ID | Model Implementations & Responsibilities | Git Branch |
|---|---|---|---|
| **Member 1** | `CIT-24-01-0427` | **Logistic Regression** & **LSTM** (Baseline ML & Deep RNN) | `feature/CIT-24-01-0427-Logistic-Regression+LSTM` |
| **Member 2** | `CIT-24-01-0363` | **SVM** & **GRU** (Kernel Classifier & Gated RNN) | `feature/CIT-24-01-0363-SVM+GRU` |
| **Member 3** | `CIT-24-01-0238` | **Random Forest** & **BERT** (Ensemble Bagging & Transformer) | `feature/CIT-24-01-0238-Random-Forest+BERT` |

---

## 🎯 Problem Statement

In the modern digital media era, the rapid proliferation of fake news, misinformation, and sensationalized articles presents a severe threat to public trust, social stability, and informed decision-making. 

Manual verification of news articles by human fact-checkers is slow, expensive, and unscalable given the millions of articles published daily online. 

**Objective:** Develop an automated, scalable, and highly accurate NLP classification system. The system leverages advanced feature engineering (TF-IDF), sequential deep neural networks (LSTM, GRU), and pre-trained language models (BERT) to detect fake news articles and present predictions through a user-friendly Streamlit web interface.

---

## 📊 Dataset Information

### Primary Dataset (ISOT Fake News Dataset)
- **Raw Articles**:
  - `Fake.csv`: 23,481 fabricated or unverified articles.
  - `True.csv`: 21,417 factual articles (sourced primarily from Reuters).
  - *Total Raw Dataset Size*: 44,898 news articles.

### Preprocessing & Leakage Prevention Pipeline
1. **Encoding & HTML Repair**: Unescaped HTML entities and fixed corrupted text encodings (`ftfy` / Unicode NFKC normalization).
2. **Dateline & Artifact Removal**: Stripped publisher datelines (e.g., `"WASHINGTON (Reuters) -"` or explicit `"Reuters"` tags) to eliminate data leakage and prevent models from relying on source cues rather than text semantics.
3. **Token Replacement**: Replaced external URLs with `URL` tokens and email addresses with `EMAIL` tokens.
4. **Deduplication & Conflict Resolution**: Calculated cryptographic text hashes (`_duplicate_key`) to prune exact duplicates and remove conflicting records assigned opposite labels.
5. **Cleaned Dataset (`cleaned_dataset.csv`)**: 27,832 balanced records (50% Fake, 50% Real).
6. **Data Partitioning**: Stratified 80% training set (~22,265 records) and 20% untouched test set (~5,567 records).

### External Validation Dataset
- **PolyglotFakeFacts v2.0**: 6,762 external cross-domain news articles used to evaluate real-world model generalization outside the training distribution.

---

## 🛠️ Setup Instructions

### Prerequisites
- **Python 3.9+** installed on your system.
- **Git** for version control.

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Aravinda123-arch/NLP_Group_32.git
   cd Fake-News-Detection
   ```

2. **Create and Activate a Virtual Environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Dataset Structure:**
   Ensure `Fake.csv` and `True.csv` are placed inside the `data/` directory:
   ```
   project-root/
   ├── data/
   │   ├── Fake.csv
   │   └── True.csv
   ```

---

## 🚀 How to Run the Project

### 1. Data Cleaning & Preprocessing
Execute the preprocessing scripts to clean raw datasets and prepare model inputs:
```bash
python src/clean_data.py
python src/Dataset_clean.py
python src/advanced_preprocess.py
```

### 2. Feature Engineering & Dataset Splitting
Split the data and compute TF-IDF vector representations:
```bash
python src/data_split.py
python src/TF-IDF_fearure_engineering.py
```

### 3. Model Training & Evaluation

- **Random Forest (Ensemble Model)**:
  ```bash
  python src/Random_forest_training.py
  python src/random_forest_internal_test.py
  python src/random_forest_external_test.py
  ```

- **BERT (Transformer Model)**:
  ```bash
  python src/bert_prepare_data.py
  python src/bert_training.py
  python src/bert_internal_test.py
  python src/bert_external_test.py
  ```

### 4. Launch the Interactive Web Application
Run the Streamlit app to interactively test news articles against all member models:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🤖 Model Summary

The project evaluates 6 distinct models spanning classic machine learning, recurrent neural networks, ensemble learning, and transformer architectures:

| Model | Model Type | Feature Representation / Architecture Details |
|---|---|---|
| **Logistic Regression** | Linear Classifier | Uses TF-IDF n-gram features (5,000 max features). Fast baseline classifier providing interpretable weights. |
| **LSTM** | Recurrent Neural Network | Deep learning network capturing sequential dependencies and contextual relationships across long sentence structures. |
| **SVM** | Kernel Classifier | Support Vector Machine optimized with linear/RBF kernel on TF-IDF features to discover optimal decision hyperplanes. |
| **GRU** | Gated Recurrent Network | Gated RNN architecture providing faster training and lower computational footprint than standard LSTM while retaining sequence memory. |
| **Random Forest** | Ensemble Classifier | 300 decision trees trained with balanced sub-sampling and square-root feature split rules to prevent overfitting. |
| **Fine-Tuned BERT** | Transformer (`bert-base-uncased`) | Bidirectional Transformer encoder fine-tuned over 3 epochs with max sequence length 256, AdamW optimizer (`lr=2e-5`), and linear warmup. |

---

## 📈 Results Summary

### Internal Test Performance (Cleaned ISOT Test Set)

| Member | Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Member 1** | Logistic Regression | **93.20%** | 93.10% | 93.20% | 93.10% | 0.9780 |
| **Member 1** | LSTM | **96.50%** | 96.40% | 96.50% | 96.40% | 0.9910 |
| **Member 2** | SVM | **94.80%** | 94.70% | 94.80% | 94.70% | 0.9850 |
| **Member 2** | GRU | **95.90%** | 95.80% | 95.90% | 95.80% | 0.9890 |
| **Member 3** | Random Forest | **98.74%** | 98.74% | 98.74% | 98.74% | 0.9990 |
| **Member 3** | **Fine-Tuned BERT** | **99.96%** | **99.96%** | **99.96%** | **99.96%** | **1.0000** |

*Note: Fine-tuned BERT achieved top internal accuracy (99.96%, 95% Confidence Interval: 99.89% - 100.00%).*

### External Cross-Domain Performance (PolyglotFakeFacts v2.0 Dataset)

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 55.97% | 55.62% | 55.85% | 55.34% | 0.5836 |
| **Fine-Tuned BERT** | 63.04% | 60.88% | 60.11% | 60.22% | 0.6410 |

*Key Finding:* Evaluating models on external datasets demonstrates the impact of domain shift in real-world news detection, emphasizing the essential role of dateline removal and broad domain training data.

---

## 📁 Repository Structure

```
Fake-News-Detection/
├── data/                       # Raw and processed datasets (Fake.csv, True.csv, cleaned_dataset.csv)
├── models/                     # Saved model artifacts (Random Forest .pkl, BERT checkpoints)
├── notebooks/                  # Model Jupyter notebooks per member branch
│   ├── BERT.ipynb
│   ├── RandomForest.ipynb
│   └── ...
├── reports/                    # Generated evaluation reports, confusion matrices & metrics CSVs
├── screenshots/                # Application UI screenshots
├── src/                        # Modular Python pipeline scripts
│   ├── Dataset_clean.py        # Advanced dataset cleaning & dateline removal
│   ├── TF-IDF_fearure_engineering.py # TF-IDF feature extraction
│   ├── Random_forest_training.py
│   ├── bert_training.py        # PyTorch BERT fine-tuning
│   ├── bert_internal_test.py   # Internal evaluation script
│   ├── bert_external_test.py   # External cross-domain evaluation
│   └── app.py / predict_real_world_news.py
├── videos/                     # Demo recordings
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Project Python dependencies
└── README.md                   # Project documentation
```
