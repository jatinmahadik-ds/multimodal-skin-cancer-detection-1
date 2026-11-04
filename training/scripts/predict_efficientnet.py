import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"E:\Final year project\skin_cancer_project\training\models\efficientnet_b0_best.keras"

# Change this to the image you want to test
IMAGE_PATH = r"E:\Final year project\ISIC 2019 JPG 224x224 RESIZED dataset\dataset_split\train\cancer\ISIC_0000002.jpg"
IMG_SIZE = 224


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading EfficientNetB0 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image:")
print(IMAGE_PATH)

img = image.load_img(
    IMAGE_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb"
)

img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# EfficientNet preprocessing
img_array = preprocess_input(img_array)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(img_array, verbose=0)

probability = float(prediction[0][0])


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("EFFICIENTNETB0 PREDICTION")
print("=" * 60)

print(f"Cancer probability: {probability:.4f}")
print(f"Cancer percentage:   {probability * 100:.2f}%")

if probability >= 0.5:
    print("\nPrediction: CANCER")
else:
    print("\nPrediction: NON-CANCER")

print("=" * 60)