import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings("ignore")


IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-4
DATA_DIR = r"C:\Users\UG\chest_xray"     
RESULTS_DIR   = "./results"
CLASSES       = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES   = 2
RANDOM_SEED   = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


print("\n[INFO] Setting up data generators ...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.1         
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=RANDOM_SEED
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=RANDOM_SEED
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print(f"  Train samples : {train_gen.samples}")
print(f"  Val   samples : {val_gen.samples}")
print(f"  Test  samples : {test_gen.samples}")

def evaluate_and_plot(y_true, y_pred, approach_name, save_prefix):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"  {approach_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=CLASSES)}")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"Confusion Matrix – {approach_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{save_prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


print("\n\n[APPROACH 1] ResNet50 Feature Extractor + SVM")
print("-" * 50)

feature_extractor = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",               
    input_shape=(*IMG_SIZE, 3)
)
feature_extractor.trainable = False   

def extract_features(generator, model, n_steps=None):
    """Extract deep features for all images in a generator."""
    features, labels = [], []
    steps = n_steps or len(generator)
    for i, (X_batch, y_batch) in enumerate(generator):
        if i >= steps:
            break
        feat = model.predict(X_batch, verbose=0)
        features.append(feat)
        labels.append(y_batch)
    return np.vstack(features), np.concatenate(labels)

print("  Extracting train features ...")
t0 = time.time()
train_gen_fe = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)
X_train_feat, y_train = extract_features(train_gen_fe, feature_extractor)

print("  Extracting test features ...")
test_gen_fe = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)
X_test_feat, y_test = extract_features(test_gen_fe, feature_extractor)
feat_time = time.time() - t0
print(f"  Feature extraction time : {feat_time:.1f}s")
print(f"  Feature vector shape    : {X_train_feat.shape[1]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled  = scaler.transform(X_test_feat)

print("  Training SVM (linear kernel) ...")
t1 = time.time()
svm = SVC(kernel="linear", C=1.0, random_state=RANDOM_SEED, probability=True)
svm.fit(X_train_scaled, y_train)
svm_train_time = time.time() - t1
print(f"  SVM training time : {svm_train_time:.1f}s")

y_pred_svm = svm.predict(X_test_scaled)
metrics_app1 = evaluate_and_plot(y_test, y_pred_svm,
                                  "Approach 1: ResNet50 + SVM",
                                  "approach1_svm")
metrics_app1["train_time"] = feat_time + svm_train_time


print("\n\n[APPROACH 2] End-to-End ResNet50 Fine-Tuned Classifier")
print("-" * 50)

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)  

e2e_model = models.Model(inputs=base_model.input, outputs=output)
e2e_model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
    ModelCheckpoint(f"{RESULTS_DIR}/best_e2e_resnet50.keras", save_best_only=True)
]

print("  Phase 1 – Training new head (base frozen) ...")
t2 = time.time()
history1 = e2e_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

print("\n  Phase 2 – Fine-tuning top 30 ResNet50 layers ...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

e2e_model.compile(
    optimizer=optimizers.Adam(learning_rate=LR / 10),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = e2e_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
e2e_train_time = time.time() - t2
print(f"  Total E2E training time : {e2e_train_time:.1f}s")

def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

history = merge_histories(history1, history2)

test_gen.reset()
y_pred_prob = e2e_model.predict(test_gen, verbose=0)
y_pred_e2e  = (y_pred_prob.flatten() >= 0.5).astype(int)
y_true_e2e  = test_gen.classes

metrics_app2 = evaluate_and_plot(y_true_e2e, y_pred_e2e,
                                  "Approach 2: End-to-End ResNet50",
                                  "approach2_e2e")
metrics_app2["train_time"] = e2e_train_time


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["accuracy"], label="Train Acc")
axes[0].plot(history["val_accuracy"], label="Val Acc")
axes[0].set_title("Accuracy – End-to-End ResNet50", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["loss"], label="Train Loss")
axes[1].plot(history["val_loss"], label="Val Loss")
axes[1].set_title("Loss – End-to-End ResNet50", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/approach2_learning_curves.png", dpi=150)
plt.close()
print(f"\n[INFO] Learning curves saved.")

metrics_keys  = ["accuracy", "precision", "recall", "f1"]
metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

app1_vals = [metrics_app1[k] for k in metrics_keys]
app2_vals = [metrics_app2[k] for k in metrics_keys]

x_pos = np.arange(len(metrics_keys))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x_pos - width/2, app1_vals, width,
               label="Approach 1: ResNet50 + SVM", color="#4C8BBF", edgecolor="white")
bars2 = ax.bar(x_pos + width/2, app2_vals, width,
               label="Approach 2: End-to-End ResNet50", color="#E07B54", edgecolor="white")

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Comparative Performance: Approach 1 vs Approach 2\n(Chest X-Ray Pneumonia Dataset – ResNet50)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/comparative_bar_chart.png", dpi=150)
plt.close()

print(f"\n[INFO] Comparative bar chart saved.")

fig, ax = plt.subplots(figsize=(6, 4))
times = [metrics_app1["train_time"], metrics_app2["train_time"]]
colors = ["#4C8BBF", "#E07B54"]
bars = ax.bar(["Approach 1\n(ResNet50+SVM)", "Approach 2\n(End-to-End)"], times, color=colors, width=0.45)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{val:.0f}s", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Training Time (seconds)", fontsize=11)
ax.set_title("Training Time Comparison", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_time_comparison.png", dpi=150)
plt.close()

print("\n\n" + "="*60)
print("  FINAL SUMMARY – CHEST X-RAY (PNEUMONIA)")
print("="*60)
print(f"  {'Metric':<15} {'App1 SVM':>12} {'App2 E2E':>12}")
print("-"*42)
for k, lbl in zip(metrics_keys, metrics_labels):
    print(f"  {lbl:<15} {metrics_app1[k]:>12.4f} {metrics_app2[k]:>12.4f}")
print("-"*42)
print(f"  {'Train Time':<15} {metrics_app1['train_time']:>11.1f}s {metrics_app2['train_time']:>11.1f}s")
print("="*60)
print(f"\n[INFO] All plots saved to: {RESULTS_DIR}/")
