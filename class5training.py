import pandas as pd
import numpy as np
import joblib
import os

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("result", exist_ok=True)
os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/train_validation.csv")

y = df["ProdTaken"].astype(int).values
X = df.drop("ProdTaken", axis=1)

preprocessor = joblib.load("data/preprocessor.joblib")
X_processed = preprocessor.transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

input_dim = X_train.shape[1]

pos_frac = float(np.mean(y_train))
alpha = 1.0 - pos_frac
alpha = float(np.clip(alpha, 0.25, 0.90))

loss_fn = keras.losses.BinaryFocalCrossentropy(
    gamma=2.0,
    alpha=alpha
)

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),

    keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(2e-5)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.35),

    keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(2e-5)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),

    keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(2e-5)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.15),

    keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(2e-5)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.10),

    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn,
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=10, restore_best_weights=False
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1
)

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="data/best_model.keras",
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=128,
    callbacks=[early_stop, reduce_lr, ckpt],
    verbose=1
)

model = keras.models.load_model("data/best_model.keras")
model.save("data/model.keras")

val_probs = model.predict(X_val).ravel()

thresholds = np.linspace(0.01, 0.99, 500)
best_thr = 0.5
best_acc = -1.0

for t in thresholds:
    preds = (val_probs >= t).astype(int)
    acc = (preds == y_val).mean()
    if acc > best_acc:
        best_acc = acc
        best_thr = t

print(f"\nTrain positive rate: {pos_frac:.4f}  -> focal alpha used: {alpha:.4f}")
print(f"Best threshold (val accuracy): {best_thr:.2f}")
print(f"Best val accuracy: {best_acc:.4f}")

val_pred = (val_probs >= best_thr).astype(int)

cm = confusion_matrix(y_val, val_pred)
print("\nConfusion Matrix (Validation):")
print(cm)

print("\nClassification Report (Validation):")
print(classification_report(y_val, val_pred, digits=4))

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "val"])
plt.tight_layout()
plt.savefig("result/train_val_loss.png", dpi=200)
plt.close()

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "val"])
plt.tight_layout()
plt.savefig("result/train_val_accuracy.png", dpi=200)
plt.close()

plt.figure()
plt.plot(history.history["auc"])
plt.plot(history.history["val_auc"])
plt.title("AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend(["train", "val"])
plt.tight_layout()
plt.savefig("result/train_val_auc.png", dpi=200)
plt.close()

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (Validation)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig("result/confusion_matrix_val.png", dpi=200)
plt.close()

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("result/history.csv", index=False)
print("Saved training history to result/history.csv")

print("\nSaved graphs to result/: train_val_loss.png, train_val_accuracy.png, train_val_auc.png, confusion_matrix_val.png")
print("Saved best model to: data/best_model.keras")
print("Saved final model to: data/model.keras")