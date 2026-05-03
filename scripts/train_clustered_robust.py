import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = r'D:\Code\Dataset_clustered'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30
CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']

def get_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'), shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'validation'), shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'test'), shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )
    return train_ds, val_ds, test_ds

# PHASE 4: ADVANCED AUGMENTATION PIPELINE
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    layers.RandomBrightness(factor=0.3),
    layers.RandomContrast(factor=0.3),
    layers.GaussianNoise(stddev=0.1) # Simulate sensor noise implicitly
])

def build_model():
    # Phase 5: MobileNetV3Small with L2 regularization and Dropout
    base_model = MobileNetV3Small(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    
    # Unfreeze the top layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    # L2 regularization to prevent memorization
    outputs = layers.Dense(len(CLASSES), activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x)

    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def add_noise_blur(image):
    # Dirty test simulation (Phase 6)
    img_arr = image.numpy()
    if np.random.rand() > 0.5:
        img_arr = cv2.GaussianBlur(img_arr, (5, 5), 0)
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 25, img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0, 255)
    return img_arr

if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    model = build_model()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(filepath=r'D:\Code\models\rice_disease_cluster_robust.keras', monitor='val_accuracy', save_best_only=True)
    ]
    
    print("\nStarting PHASE 5: Model Training (Generalization First)")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    
    print("\nStarting PHASE 6: Honest Evaluation")
    # Clean Eval
    clean_loss, clean_acc = model.evaluate(test_ds)
    print(f"Clean Test Accuracy: {clean_acc*100:.2f}%")
    
    # Dirty Eval
    print("\nRunning Dirty Data Robustness Test...")
    y_true, y_pred_dirty = [], []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        dirty_images = np.array([add_noise_blur(img) for img in images])
        y_pred_dirty.extend(np.argmax(model.predict(dirty_images, verbose=0), axis=1))

    y_true = np.array(y_true)
    y_pred_dirty = np.array(y_pred_dirty)
    dirty_acc = np.mean(y_true == y_pred_dirty) * 100

    print(f"Dirty Test Accuracy: {dirty_acc:.2f}%")
    print(classification_report(y_true, y_pred_dirty, target_names=CLASSES, digits=4))