import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings("ignore")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
DATA_DIR = r"C:\Users\UG\chest_xray"
RESULTS_DIR = "./results"
CLASSES     = ["NORMAL", "PNEUMONIA"]
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plain_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
aug_datagen   = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.1
)

train_gen = aug_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="training", shuffle=True, seed=RANDOM_SEED
)
val_gen = aug_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="validation", shuffle=False, seed=RANDOM_SEED
)

def evaluate_model(y_true, y_pred, name, prefix):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(f"  Accuracy : {acc:.4f}  Precision : {prec:.4f}")
    print(f"  Recall   : {rec:.4f}  F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES)}")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"CM – {name}", fontweight="bold")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{prefix}_cm.png", dpi=150)
    plt.close()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


print("\n[BONUS] EfficientNet-B0 – Approach 1: Feature Extractor + SVM")

eff_extractor = EfficientNetB0(
    weights="imagenet", include_top=False,
    pooling="avg", input_shape=(*IMG_SIZE, 3)
)
eff_extractor.trainable = False

def extract(gen, model):
    feats, labs = [], []
    for i, (X, y) in enumerate(gen):
        if i >= len(gen): break
        feats.append(model.predict(X, verbose=0))
        labs.append(y)
    return np.vstack(feats), np.concatenate(labs)

train_noshuf = plain_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)
test_noshuf = plain_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)

t0 = time.time()
X_tr, y_tr = extract(train_noshuf, eff_extractor)
X_te, y_te = extract(test_noshuf, eff_extractor)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

svm = SVC(kernel="linear", C=1.0, random_state=RANDOM_SEED)
svm.fit(X_tr_s, y_tr)
app1_time = time.time() - t0

y_pred1 = svm.predict(X_te_s)
m1 = evaluate_model(y_te, y_pred1, "EfficientNet-B0 + SVM", "effnet_app1")
m1["train_time"] = app1_time


print("\n[BONUS] EfficientNet-B0 – Approach 2: End-to-End")

base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(1, activation="sigmoid")(x)
eff_model = models.Model(base.input, out)

eff_model.compile(optimizer=optimizers.Adam(LR),
                  loss="binary_crossentropy", metrics=["accuracy"])

cbs = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3),
    ModelCheckpoint(f"{RESULTS_DIR}/best_effnet.keras", save_best_only=True)
]

t1 = time.time()
hist1 = eff_model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=cbs, verbose=1)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
eff_model.compile(optimizer=optimizers.Adam(LR / 10),
                  loss="binary_crossentropy", metrics=["accuracy"])
hist2 = eff_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cbs, verbose=1)
app2_time = time.time() - t1

test_noshuf.reset()
y_prob2 = eff_model.predict(test_noshuf, verbose=0).flatten()
y_pred2 = (y_prob2 >= 0.5).astype(int)
m2 = evaluate_model(test_noshuf.classes, y_pred2, "EfficientNet-B0 E2E", "effnet_app2")
m2["train_time"] = app2_time

all_acc = hist1.history["accuracy"] + hist2.history["accuracy"]
all_val = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
all_loss = hist1.history["loss"] + hist2.history["loss"]
all_vloss = hist1.history["val_loss"] + hist2.history["val_loss"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(all_acc, label="Train Acc"); axes[0].plot(all_val, label="Val Acc")
axes[0].set_title("Accuracy – EfficientNet-B0 E2E", fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(all_loss, label="Train Loss"); axes[1].plot(all_vloss, label="Val Loss")
axes[1].set_title("Loss – EfficientNet-B0 E2E", fontweight="bold")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/effnet_learning_curves.png", dpi=150)
plt.close()

keys = ["accuracy", "precision", "recall", "f1"]
labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
x_pos = np.arange(len(keys))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x_pos - width/2, [m1[k] for k in keys], width,
            label="EfficientNet-B0 + SVM", color="#7C5CBF", edgecolor="white")
b2 = ax.bar(x_pos + width/2, [m2[k] for k in keys], width,
            label="EfficientNet-B0 E2E", color="#5CBF8A", edgecolor="white")
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", fontsize=8.5)
ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title("EfficientNet-B0 – Approach 1 vs Approach 2\n(Chest X-Ray Pneumonia)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/effnet_comparative_bar.png", dpi=150)
plt.close()

print("\n\n" + "="*60)
print("  BONUS SUMMARY – EfficientNet-B0 (Chest X-Ray)")
print("="*60)
print(f"  {'Metric':<15} {'Eff+SVM':>12} {'Eff E2E':>12}")
print("-"*42)
for k, lbl in zip(keys, labels):
    print(f"  {lbl:<15} {m1[k]:>12.4f} {m2[k]:>12.4f}")
print("-"*42)
print(f"  {'Train Time':<15} {m1['train_time']:>11.1f}s {m2['train_time']:>11.1f}s")
print("="*60)
