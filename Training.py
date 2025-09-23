import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

print("Available devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device.device_type}: {device.name}")

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth enabled")

# Paths
train_images_path = 'resources/augmented_train/images'
train_labels_path = 'resources/augmented_train/labels'
val_images_path = 'resources/valid/images'
val_labels_path = 'resources/valid/labels'

# Logger
log_data = CSVLogger("Log_data.csv", append=True)

# --- Decode image ---
def decode_image(image_path):
    img_bytes = tf.io.read_file(image_path)
    if tf.strings.regex_full_match(image_path, ".*\\.png"):
        image = tf.image.decode_png(img_bytes, channels=3)
    else:
        image = tf.image.decode_jpeg(img_bytes, channels=3)
    return image

# --- YOLO label to crop + resize ---
def process_resize(img_path, lab_path):
    image = decode_image(img_path)
    label_content = tf.io.read_file(lab_path)
    label_values = tf.strings.split(tf.strings.strip(label_content))
    
    # fallback for invalid label
    if tf.size(label_values) < 5:
        image = tf.image.resize(image, [96, 96]) / 255.0
        class_id = tf.cond(tf.size(label_values) > 0,
                           lambda: tf.cast(tf.strings.to_number(label_values[0], tf.float32), tf.int32),
                           lambda: tf.constant(0, tf.int32))
        return image, class_id
    
    # class ID
    class_id = tf.cast(tf.strings.to_number(label_values[0], tf.float32), tf.int32)
    
    # YOLO coordinates
    yolo_coords = tf.strings.to_number(label_values[1:5], tf.float32)
    yolo_coords = tf.ensure_shape(yolo_coords, [4])
    
    x_c, y_c, w, h = tf.split(yolo_coords, num_or_size_splits=4)
    x_c, y_c, w, h = tf.squeeze(x_c), tf.squeeze(y_c), tf.squeeze(w), tf.squeeze(h)
    
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)
    
    x_min = tf.maximum(0.0, (x_c - w/2) * img_w)
    y_min = tf.maximum(0.0, (y_c - h/2) * img_h)
    x_max = tf.minimum(img_w, (x_c + w/2) * img_w)
    y_max = tf.minimum(img_h, (y_c + h/2) * img_h)
    
    cropped = tf.image.crop_to_bounding_box(
        image,
        tf.cast(y_min, tf.int32),
        tf.cast(x_min, tf.int32),
        tf.cast(y_max - y_min, tf.int32),
        tf.cast(x_max - x_min, tf.int32)
    )
    
    cropped = tf.image.resize(cropped, [96, 96]) / 255.0
    return cropped, class_id

# --- Dataset creation ---
def create_dataset(image_dir, label_dir, batch_size=32):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    dataset = dataset.map(process_resize, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_images_path, train_labels_path)
val_dataset = create_dataset(val_images_path, val_labels_path)

# --- Class weights ---
train_labels_list = []
for label_file in os.listdir(train_labels_path):
    if not label_file.endswith('.txt'):
        continue
    path = os.path.join(train_labels_path, label_file)
    with open(path, 'r') as f:
        line = f.readline().strip()
        if line:
            try:
                cls = int(float(line.split()[0]))
                train_labels_list.append(cls)
            except:
                continue

train_labels_arr = np.array(train_labels_list, dtype=int)
class_weights_train = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_labels_arr),
    y=train_labels_arr
)
class_weight_dict = dict(enumerate(class_weights_train))


def residual_block(x, filters, stride=1):
    shortcut = x  

    x = layers.Conv2D(filters, (3,3), padding='same', strides=stride)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3,3), padding='same', strides=1)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


# --- Model architecture ---

inputs = Input(shape=(96,96,3))

x = layers.Conv2D(32, (3,3), strides=1, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = residual_block(x, filters=32)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(64, (3,3), strides=1, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = residual_block(x, filters=64)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, (3,3), strides=1, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = residual_block(x, filters=128)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.3)(x)


x = layers.Conv2D(256, (3,3), strides=1, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = residual_block(x, filters=256)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.3)(x)

x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(8, activation='softmax')(x)

# --- Create model ---
model = models.Model(inputs, outputs)


model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
)

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=40, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=10),
    ModelCheckpoint("Best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    log_data
]

# --- Training ---
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=600,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
