# Fake News Detection Project

This repository contains machine learning and deep learning models implemented by different team members to detect fake news.

## Project Structure

The project uses git branches to separate work among team members. The base structure of the repository on the `main` branch is as follows:

```
Fake-News-Detection/ (main branch)
│
├── Dataset/
│     ├── Fake.csv
│     └── True.csv
│
├── app.py
├── requirements.txt
└── README.md
```

## Member Branches

Each team member works on their respective branch containing their model implementations at the root of the project:

- **Branch `member1`**:
  - `LogisticRegression.ipynb`
  - `LSTM.ipynb`
- **Branch `member2`**:
  - `SVM.ipynb`
  - `GRU.ipynb`
- **Branch `member3`**:
  - `RandomForest.ipynb`
  - `BERT.ipynb`

## Setup Instructions

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset files (`Fake.csv` and `True.csv`) inside the `Dataset` directory.
3. Switch to your respective branch to work on your models:
   ```bash
   git checkout member1  # or member2, member3
   ```
4. Run the Streamlit web application from `main`:
   ```bash
   streamlit run app.py
   ```
