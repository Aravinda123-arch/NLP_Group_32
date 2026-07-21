import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.model_selection import train_test_split

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

tfidf = joblib.load("models/tfidf.pkl")

svm = joblib.load("models/svm_model.pkl")

X_test = tfidf.transform(X_test)

prediction = svm.predict(X_test)

probability = svm.predict_proba(X_test)[:,1]

# ----------------------------------------

print()

print("Accuracy")

print(accuracy_score(y_test,prediction))

print()

print("Precision")

print(precision_score(y_test,prediction))

print()

print("Recall")

print(recall_score(y_test,prediction))

print()

print("F1 Score")

print(f1_score(y_test,prediction))

print()

print(classification_report(y_test,prediction))

# ----------------------------------------

cm = confusion_matrix(y_test,prediction)

ConfusionMatrixDisplay(cm).plot()

plt.show()

# ----------------------------------------

fpr,tpr,_ = roc_curve(

    y_test,

    probability

)

roc_auc = auc(fpr,tpr)

plt.figure(figsize=(6,6))

plt.plot(

    fpr,

    tpr,

    label="AUC = %0.3f"%roc_auc

)

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()