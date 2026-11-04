# ============================================================
# STEP 16
# EfficientNetB0 - ISIC 2019 CPU Training
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset_split"
)

TRAIN_CSV = os.path.join(
    DATASET_DIR,
    "balanced_train.csv"
)

VAL_CSV = os.path.join(
    DATASET_DIR,
    "validation.csv"
)

TEST_CSV = os.path.join(
    DATASET_DIR,
    "test.csv"
)

# Project training output directories
TRAINING_DIR = r"E:\Final year project\skin_cancer_project\training"

MODEL_DIR = os.path.join(
    TRAINING_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    TRAINING_DIR,
    "results",
    "efficientnet"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMG_SIZE = 224

BATCH_SIZE = 16

HEAD_EPOCHS = 10

FINE_TUNE_EPOCHS = 10

LEARNING_RATE = 1e-4

FINE_TUNE_LR = 1e-5

SEED = 42


# ============================================================
# CPU CONFIGURATION
# ============================================================

# Use CPU only.
tf.config.set_visible_devices([], "GPU")

print("\n==========================================")
print("EFFICIENTNETB0 CPU TRAINING")
print("==========================================")

print("TensorFlow version:", tf.__version__)

print("Physical CPUs:")
print(tf.config.list_physical_devices("CPU"))

print("GPU disabled: YES")


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD CSV FILES
# ============================================================

print("\n==========================================")
print("LOADING DATA")
print("==========================================")


train_df = pd.read_csv(TRAIN_CSV)

val_df = pd.read_csv(VAL_CSV)

test_df = pd.read_csv(TEST_CSV)


print("\nTrain rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["isic_id", "target"]

for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name}.csv is missing columns: {missing}"
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTrain distribution:")
print(train_df["target"].value_counts().sort_index())

print("\nValidation distribution:")
print(val_df["target"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["target"].value_counts().sort_index())


# ============================================================
# 3. CREATE IMAGE PATHS
# ============================================================

def create_image_path(row, split):

    image_id = str(row["isic_id"])

    target = int(row["target"])

    if target == 1:

        class_name = "cancer"

    elif target == 0:

        class_name = "non_cancer"

    else:

        raise ValueError(
            f"Invalid target: {target}"
        )

    return os.path.join(
        DATASET_DIR,
        split,
        class_name,
        image_id + ".jpg"
    )


train_paths = [
    create_image_path(row, "train")
    for _, row in train_df.iterrows()
]

val_paths = [
    create_image_path(row, "validation")
    for _, row in val_df.iterrows()
]

test_paths = [
    create_image_path(row, "test")
    for _, row in test_df.iterrows()
]


train_labels = train_df["target"].astype(
    np.float32
).values

val_labels = val_df["target"].astype(
    np.float32
).values

test_labels = test_df["target"].astype(
    np.float32
).values


# ============================================================
# 4. VERIFY FILES
# ============================================================

print("\nChecking image paths...")

missing_train = [
    p for p in train_paths
    if not os.path.exists(p)
]

missing_val = [
    p for p in val_paths
    if not os.path.exists(p)
]

missing_test = [
    p for p in test_paths
    if not os.path.exists(p)
]


print("Missing training images:", len(missing_train))
print("Missing validation images:", len(missing_val))
print("Missing test images:", len(missing_test))


if missing_train or missing_val or missing_test:

    raise FileNotFoundError(
        "Some image files are missing."
    )


# ============================================================
# 5. TF.DATA IMAGE LOADER
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ------------------------------------------------------------
# TRAIN DATASET
# ------------------------------------------------------------

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_paths),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)


# ------------------------------------------------------------
# VALIDATION DATASET
# ------------------------------------------------------------

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    AUTOTUNE
)


# ------------------------------------------------------------
# TEST DATASET
# ------------------------------------------------------------

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

augmentation = keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),

    ],
    name="medical_image_augmentation"
)


# ============================================================
# 7. BUILD EFFICIENTNETB0
# ============================================================

print("\n==========================================")
print("BUILDING EFFICIENTNETB0")
print("==========================================")


base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# Freeze pretrained base initially

base_model.trainable = False


inputs = keras.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),
    name="skin_image"
)


# Augmentation only during training

x = augmentation(inputs)


# EfficientNet

x = base_model(
    x,
    training=False
)


# Global feature vector

x = layers.GlobalAveragePooling2D()(x)


# Dropout

x = layers.Dropout(
    0.30
)(x)


# Binary classifier

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="cancer_probability"
)(x)


model = keras.Model(
    inputs,
    outputs,
    name="EfficientNetB0_ISIC2019"
)


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        keras.metrics.Precision(
            name="precision"
        ),

        keras.metrics.Recall(
            name="recall"
        ),

        keras.metrics.AUC(
            name="auc"
        )
    ]
)


model.summary()


# ============================================================
# 9. CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "efficientnet_b0_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "efficientnet_b0_final.keras"
)


callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    keras.callbacks.CSVLogger(
        os.path.join(
            RESULTS_DIR,
            "training_history.csv"
        )
    )
]


# ============================================================
# 10. TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n==========================================")
print("PHASE 1 - TRAIN CLASSIFICATION HEAD")
print("==========================================")


history_head = model.fit(
    train_dataset,

    validation_data=val_dataset,

    epochs=HEAD_EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# 11. FINE-TUNING
# ============================================================

print("\n==========================================")
print("PHASE 2 - FINE-TUNING")
print("==========================================")


# Unfreeze base model

base_model.trainable = True


# Freeze most layers.
# Fine-tune only the final 30 layers.

for layer in base_model.layers[:-30]:

    layer.trainable = False


# Keep BatchNorm layers frozen for stability

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LR
    ),

    loss="binary_crossentropy",

    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        keras.metrics.Precision(
            name="precision"
        ),

        keras.metrics.Recall(
            name="recall"
        ),

        keras.metrics.AUC(
            name="auc"
        )
    ]
)


history_fine = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# 12. SAVE FINAL MODEL
# ============================================================

model.save(
    final_model_path
)

print("\nFinal model saved:")
print(final_model_path)

print("\nBest model saved:")
print(best_model_path)


# ============================================================
# 13. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = keras.models.load_model(
    best_model_path
)


# ============================================================
# 14. TEST EVALUATION
# ============================================================

print("\n==========================================")
print("FINAL TEST EVALUATION")
print("==========================================")


test_results = best_model.evaluate(
    test_dataset,
    verbose=1
)


print("\nKeras test metrics:")

for name, value in zip(
    best_model.metrics_names,
    test_results
):

    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# 15. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = best_model.predict(
    test_dataset,
    verbose=1
).ravel()


predictions = (
    probabilities >= 0.5
).astype(int)


true_labels = test_labels.astype(int)


# ============================================================
# 16. METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    true_labels,
    probabilities
)


print("\n==========================================")
print("TEST RESULTS")
print("==========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    true_labels,

    predictions,

    target_names=[
        "Non-cancer",
        "Cancer"
    ],

    digits=4,

    zero_division=0
)


print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(report)


with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions
)


print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm
)

plt.title(
    "EfficientNetB0 Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    [
        "Non-cancer",
        "Cancer"
    ]
)

plt.yticks(
    [0, 1],
    [
        "Non-cancer",
        "Cancer"
    ]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),

    dpi=200
)

plt.close()


# ============================================================
# 19. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = test_df.copy()

prediction_df[
    "predicted_probability"
] = probabilities

prediction_df[
    "predicted_target"
] = predictions


prediction_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "test_predictions.csv"
    ),

    index=False
)


# ============================================================
# 20. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ],

    "value": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]

})


summary.to_csv(

    os.path.join(
        RESULTS_DIR,
        "test_metrics.csv"
    ),

    index=False
)


print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print(
    "\nBest model:"
)

print(
    best_model_path
)

print(
    "\nResults directory:"
)

print(
    RESULTS_DIR
)