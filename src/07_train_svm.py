import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# ----------------------------------------

df = pd.read_csv("data/stemmed_news.csv")

X = df["processed_text"]

y = df["label"]

# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ----------------------------------------

tfidf = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1,2)

)

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

# ----------------------------------------

svm = SVC(

    kernel="linear",

    C=1,

    probability=True

)

svm.fit(X_train, y_train)

# ----------------------------------------

prediction = svm.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print()

print("SVM Accuracy")

print(accuracy)

# Save Model

joblib.dump(svm, "models/svm_model.pkl")

joblib.dump(tfidf, "models/tfidf.pkl")

print()

print("Model Saved Successfully")
