from numpy.testing import verbose
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report

val_dir = "data/processed/isimand_dataset/val"
model_path = "ml/models/cnn_model.h5"

image_size = (224,224)
batch_size = 32

class_names = ["no_damage","moderate_damage","severe_damage"]

print("[INFO] Loading model from {model_path}")
model = tf.keras.models.load_model(model_path)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
    label_mode="int"
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
val_ds = val_ds.map(lambda x,y: (normalization_layer(x),y))

y_true = []
y_pred = []

for images,labels in val_ds:
    predictions = model.predict(images,verbose=0)
    prediction_classes = np.argmax(predictions,axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(prediction_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n[INFO] Confusion Matrix:")
cm = confusion_matrix(y_true,y_pred)
print(cm)

print("\n[INFO] Classification Report:")
print(classification_report(y_true,y_pred,target_names=class_names,digits=4))
    