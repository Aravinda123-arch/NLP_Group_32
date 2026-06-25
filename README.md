# Fake News Detection Project

This repository contains machine learning and deep learning models implemented by different team members to detect fake news.

## Required Repository Structure

The project has the following base repository structure on the `main` branch:

```
project-root/
├── data/
│     ├── Fake.csv
│     └── True.csv
├── notebooks/
├── src/
├── models/
├── reports/
├── screenshots/
├── videos/
├── requirements.txt
├── README.md
└── .gitignore
```

## Member Branches

Each team member works on their respective branch. In their branch, their specific model notebooks are stored under the `notebooks/` directory:

- **Branch `member1`**:
  - `notebooks/LogisticRegression.ipynb`
  - `notebooks/LSTM.ipynb`
- **Branch `member2`**:
  - `notebooks/SVM.ipynb`
  - `notebooks/GRU.ipynb`
- **Branch `member3`**:
  - `notebooks/RandomForest.ipynb`
  - `notebooks/BERT.ipynb`

## Setup Instructions


1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset files (`Fake.csv` and `True.csv`) inside the `data/` directory.
3. Switch to your respective branch to work on your models:
   ```bash
   git checkout member1  # or member2, member3
   ```
4. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```
