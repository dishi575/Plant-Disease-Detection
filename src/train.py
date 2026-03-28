import os
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 🔹 Download dataset automatically
dataset_path = kagglehub.dataset_download("emmarex/plantdisease")
print("Dataset downloaded at:", dataset_path)

# ⚠ IMPORTANT: actual images are inside 'PlantVillage'
DATASET_PATH = os.path.join(dataset_path, "PlantVillage")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class labels for prediction
os.makedirs("models", exist_ok=True)
class_names = list(train_data.class_indices.keys())
np.save("models/classes.npy", class_names)

# Load pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Save model
model.save("models/model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig("models/accuracy.png")

print("Training complete. Model + classes saved.")