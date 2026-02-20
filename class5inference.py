import pandas as pd
import numpy as np
import joblib
import os

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("data/output", exist_ok=True)

print("Loading trained model...")

model = keras.models.load_model("data/model.keras", compile=False)

def combined_bce_l1_weights_loss(model, alpha=1.0, beta=1e-6):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_loss = tf.reduce_mean(bce)
        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
        return alpha * bce_loss + beta * l1_reg
    return loss

custom_loss = combined_bce_l1_weights_loss(model, alpha=1.0, beta=1e-6)

model.compile(
    optimizer="adam",
    loss=custom_loss,
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

print("Model loaded successfully!")
model.summary()

print("\nLoading test data...")
df_test = pd.read_csv("data/test.csv")

y_test = df_test["ProdTaken"].astype(int).values
X_test = df_test.drop("ProdTaken", axis=1)

print(f"Test dataset shape: {df_test.shape}")
print(f"\nTarget distribution:\n{pd.Series(y_test).value_counts()}")

print("\nLoading preprocessor...")
preprocessor = joblib.load("data/preprocessor.joblib")
X_test_processed = preprocessor.transform(X_test)
print(f"Features after preprocessing: {X_test_processed.shape[1]}")

print("\nMaking predictions...")
y_prob = model.predict(X_test_processed, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)

try:
    auc = roc_auc_score(y_test, y_prob)
except:
    auc = None

print("\n" + "=" * 60)
print("INFERENCE RESULTS (CLASS 5)")
print("=" * 60)
print(f"\nAccuracy: {acc:.4f}")
if auc is not None:
    print(f"AUC Score: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"})
plt.title("Confusion Matrix - Ass5 Classification Model", fontsize=16, fontweight="bold")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

tn, fp, fn, tp = cm.ravel()
stats_text = f"True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}\n\nAccuracy: {acc:.4f}"
plt.text(2.5, 0.5, stats_text, fontsize=10,
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
         verticalalignment="center")

plt.tight_layout()
plt.savefig("data/output/confusion_matrix_test.jpg", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", cbar_kws={"label": "Percentage"})
plt.title("Normalized Confusion Matrix - Ass5 Classification Model", fontsize=16, fontweight="bold")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("data/output/confusion_matrix_test_normalized.jpg", dpi=300, bbox_inches="tight")
plt.close()

results_df = df_test.copy()
results_df["predicted_label"] = y_pred
results_df["prediction_probability"] = y_prob
results_df["correct_prediction"] = (results_df["ProdTaken"].astype(int) == y_pred)
results_df.to_csv("data/output/predictions.csv", index=False)

print("\nSaved:")
print("data/output/confusion_matrix_test.jpg")
print("data/output/confusion_matrix_test_normalized.jpg")
print("data/output/predictions.csv")