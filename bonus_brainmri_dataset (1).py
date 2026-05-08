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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings("ignore")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
DATA_DIR = r"C:\Users\UG\brain_mri"
RESULTS_DIR = "./results"
RANDOM_SEED = 42
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = 4

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

aug_datagen   = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15,
    validation_split=0.15
)
plain_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = aug_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Training"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training",
    shuffle=True, seed=RANDOM_SEED
)
val_gen = aug_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Training"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation",
    shuffle=False, seed=RANDOM_SEED
)
test_gen = plain_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Testing"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

print(f"[INFO] Brain MRI: {train_gen.samples} train | {val_gen.samples} val | {test_gen.samples} test")
print(f"[INFO] Classes: {list(train_gen.class_indices.keys())}")

def evaluate_multi(y_true, y_pred, name, prefix, class_names):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(f"  Accuracy : {acc:.4f}  Precision : {prec:.4f}")
    print(f"  Recall   : {rec:.4f}  F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"CM – {name}", fontweight="bold")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{prefix}_cm.png", dpi=150)
    plt.close()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


print("\n[BONUS DS] Brain MRI – ResNet50 Feature Extractor + SVM")

extractor = ResNet50(weights="imagenet", include_top=False,
                     pooling="avg", input_shape=(*IMG_SIZE, 3))
extractor.trainable = False

def extract_multi(gen, model):
    feats, labs = [], []
    for i, (X, y) in enumerate(gen):
        if i >= len(gen): break
        feats.append(model.predict(X, verbose=0))
        labs.append(np.argmax(y, axis=1))
    return np.vstack(feats), np.concatenate(labs)

train_noshuf = plain_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Training"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)
test_noshuf = plain_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "Testing"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

t0 = time.time()
X_tr, y_tr = extract_multi(train_noshuf, extractor)
X_te, y_te = extract_multi(test_noshuf, extractor)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

svm = SVC(kernel="linear", C=1.0, decision_function_shape="ovr", random_state=RANDOM_SEED)
svm.fit(X_tr_s, y_tr)
app1_time = time.time() - t0

y_pred1 = svm.predict(X_te_s)
class_names = list(test_noshuf.class_indices.keys())
m1 = evaluate_multi(y_te, y_pred1, "Brain MRI: ResNet50 + SVM", "brainmri_app1", class_names)
m1["train_time"] = app1_time


print("\n[BONUS DS] Brain MRI – End-to-End ResNet50")

base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
e2e = models.Model(base.input, out)

e2e.compile(optimizer=optimizers.Adam(LR),
            loss="categorical_crossentropy", metrics=["accuracy"])

cbs = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3),
    ModelCheckpoint(f"{RESULTS_DIR}/best_brainmri_e2e.keras", save_best_only=True)
]

t1 = time.time()
hist1 = e2e.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=cbs, verbose=1)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
e2e.compile(optimizer=optimizers.Adam(LR / 10),
            loss="categorical_crossentropy", metrics=["accuracy"])
hist2 = e2e.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cbs, verbose=1)
app2_time = time.time() - t1

test_noshuf.reset()
y_prob2 = e2e.predict(test_noshuf, verbose=0)
y_pred2 = np.argmax(y_prob2, axis=1)
y_true2 = test_noshuf.classes
m2 = evaluate_multi(y_true2, y_pred2, "Brain MRI: E2E ResNet50", "brainmri_app2", class_names)
m2["train_time"] = app2_time

all_acc  = hist1.history["accuracy"]  + hist2.history["accuracy"]
all_val  = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
all_loss = hist1.history["loss"]  + hist2.history["loss"]
all_vl   = hist1.history["val_loss"]  + hist2.history["val_loss"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(all_acc, label="Train Acc"); axes[0].plot(all_val, label="Val Acc")
axes[0].set_title("Accuracy – Brain MRI E2E ResNet50", fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(all_loss, label="Train Loss"); axes[1].plot(all_vl, label="Val Loss")
axes[1].set_title("Loss – Brain MRI E2E ResNet50", fontweight="bold")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/brainmri_learning_curves.png", dpi=150)
plt.close()

keys = ["accuracy", "precision", "recall", "f1"]
labels_m = ["Accuracy", "Precision", "Recall", "F1-Score"]
x_pos = np.arange(len(keys)); width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x_pos - width/2, [m1[k] for k in keys], width,
            label="ResNet50 + SVM", color="#C05C5C", edgecolor="white")
b2 = ax.bar(x_pos + width/2, [m2[k] for k in keys], width,
            label="E2E ResNet50", color="#5C8FC0", edgecolor="white")
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", fontsize=8.5)
ax.set_xticks(x_pos); ax.set_xticklabels(labels_m, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title("Brain MRI Tumor – ResNet50: Approach 1 vs Approach 2",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/brainmri_comparative_bar.png", dpi=150)
plt.close()

print("\n\n" + "="*60)
print("  BONUS DATASET SUMMARY – Brain MRI Tumor (ResNet50)")
print("="*60)
print(f"  {'Metric':<15} {'ResNet+SVM':>12} {'E2E ResNet':>12}")
print("-"*42)
for k, lbl in zip(keys, labels_m):
    print(f"  {lbl:<15} {m1[k]:>12.4f} {m2[k]:>12.4f}")
print("-"*42)
print(f"  {'Train Time':<15} {m1['train_time']:>11.1f}s {m2['train_time']:>11.1f}s")
print("="*60)
