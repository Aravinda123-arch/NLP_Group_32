import sys
import os
import subprocess

try:
    import tensorflow as tf
except ModuleNotFoundError:
    if os.environ.get("_TF_RELAUNCH_ATTEMPTED"):
        print("Error: TensorFlow is not installed in the default python environment either. Please install it.")
        sys.exit(1)

    print("TensorFlow not found in the current Python environment.")
    print("Re-launching script using the system's default 'python' command...")
    env = os.environ.copy()
    env["_TF_RELAUNCH_ATTEMPTED"] = "1"
    
    # Try the known working Python 3.12 executable
    python_exe = r"C:\Users\User\AppData\Local\Microsoft\WindowsApps\python.exe"
    if not os.path.exists(python_exe):
        python_exe = "py"
        args = ["-3.12", sys.argv[0]] + sys.argv[1:]
    else:
        args = [python_exe, sys.argv[0]] + sys.argv[1:]
        
    sys.exit(subprocess.call(args, env=env))

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Use Keras via the already-imported TensorFlow object for compatibility
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
SpatialDropout1D = tf.keras.layers.SpatialDropout1D
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
EarlyStopping = tf.keras.callbacks.EarlyStopping

# Keras / TensorFlow compatible preprocessing imports
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

def main():
    # 1. File Path Resolution Relative to src/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_path = os.path.join(project_root, "data", "preprocessed_news.csv")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_save_path = os.path.join(models_dir, "lstm_model.keras")
    tokenizer_save_path = os.path.join(models_dir, "lstm_tokenizer.pickle")

    # 2. Load Preprocessed Data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path).dropna(subset=['processed_text'])

    X = df['processed_text'].astype(str).values
    y = df['label'].values

    # 3. Tokenization & Sequence Padding
    MAX_WORDS = 20000  # Max vocabulary size
    MAX_LEN = 300      # Max sequence length

    print("Tokenizing text and padding sequences...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # 4. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape:  {X_test.shape}")

    # 5. Build Sequential LSTM Architecture
    EMBEDDING_DIM = 128

    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid') # Binary Output (0 = Fake, 1 = Real)
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 6. Early Stopping Callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    # 7. Model Training
    print("\nStarting LSTM Training...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # 8. Evaluation
    print("\nEvaluating model on Test Set...")
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n==========================================")
    print("             LSTM Evaluation              ")
    print("==========================================")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("==========================================\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)']))

    # 9. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Fake', 'Predicted Real'],
                yticklabels=['Actual Fake', 'Actual Real'])
    plt.title('LSTM - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))
    plt.close()

    # 10. Save Model Artifacts
    model.save(model_save_path)

    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nLSTM Model saved successfully at: {model_save_path}")
    print(f"Tokenizer saved successfully at:  {tokenizer_save_path}")

if __name__ == "__main__":
    main()

    