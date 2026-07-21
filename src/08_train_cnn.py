import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
except (ImportError, ModuleNotFoundError):
    from keras.src.legacy.preprocessing.text import Tokenizer  # type: ignore
    from keras.utils import pad_sequences


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# ----------------------------------------
# Load Dataset
# ----------------------------------------

df = pd.read_csv("data/stemmed_news.csv")

X = df["processed_text"].astype(str)

y = df["label"]

# ----------------------------------------
# Train Test Split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ----------------------------------------
# Tokenizer
# ----------------------------------------

MAX_WORDS = 20000

tokenizer = Tokenizer(num_words=MAX_WORDS)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# ----------------------------------------
# Padding
# ----------------------------------------

MAX_LENGTH = 300

X_train_pad = pad_sequences(

    X_train_seq,

    maxlen=MAX_LENGTH,

    padding="post"

)

X_test_pad = pad_sequences(

    X_test_seq,

    maxlen=MAX_LENGTH,

    padding="post"

)

# ----------------------------------------
# CNN Model
# ----------------------------------------

model = Sequential()

model.add(

    Embedding(

        input_dim=MAX_WORDS,

        output_dim=100,

        input_length=MAX_LENGTH

    )

)

model.add(

    Conv1D(

        filters=128,

        kernel_size=5,

        activation="relu"

    )

)

model.add(MaxPooling1D(pool_size=2))

model.add(

    Conv1D(

        filters=64,

        kernel_size=3,

        activation="relu"

    )

)

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

# ----------------------------------------

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ----------------------------------------
# Callbacks
# ----------------------------------------

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "models/cnn_model.keras",

    save_best_only=True

)

# ----------------------------------------
# Train
# ----------------------------------------

history = model.fit(

    X_train_pad,

    y_train,

    epochs=10,

    batch_size=64,

    validation_split=0.2,

    callbacks=[early_stop, checkpoint],

    verbose=1

)

# ----------------------------------------
# Evaluation
# ----------------------------------------

loss, accuracy = model.evaluate(

    X_test_pad,

    y_test,

    verbose=0

)

print()

print("CNN Accuracy :", accuracy)

# ----------------------------------------
# Save Tokenizer
# ----------------------------------------

joblib.dump(

    tokenizer,

    "models/tokenizer.pkl"

)

print()

print("CNN Model Saved Successfully")