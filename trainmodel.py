import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input, BatchNormalization, LeakyReLU, Add, GlobalAveragePooling2D
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Set CUDA directory for XLA
cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
print(f"Setting CUDA directory to: {cuda_dir}")
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_dir}"

# Check if libdevice.10.bc exists
libdevice_path = os.path.join(cuda_dir, "nvvm", "libdevice", "libdevice.10.bc")
if os.path.exists(libdevice_path):
    print(f"Found libdevice at: {libdevice_path}")
else:
    print(f"Warning: {libdevice_path} not found. XLA might not work properly.")

# Print TensorFlow info
print("\nTensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("CUDA Directory:", cuda_dir)

# Define directories
TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'

# Create DataFrame
def create_dataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = create_dataframe(TRAIN_DIR)
print(train)

test = pd.DataFrame()
test['image'], test['label'] = create_dataframe(TEST_DIR)
print(test)

# Count class distribution
class_counts = train['label'].value_counts().to_dict()
print("Class distribution:", class_counts)

# Calculate class weights manually - avoiding sklearn's compute_class_weight
total_samples = len(train)
n_classes = len(class_counts)
class_weights = {}

for i, (label, count) in enumerate(class_counts.items()):
    # Weight formula: total_samples / (n_classes * count)
    weight = total_samples / (n_classes * count)
    class_weights[i] = weight

print("Class weights:", class_weights)

# Image processing functions
def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)  # Grayscale
    img = tf.image.resize(img, [96, 96])  # Increased resolution
    img = img / 255.0  # Normalize
    return img, label

def create_dataset(df, le=None, batch_size=32, shuffle=True):
    image_paths = df['image'].values
    labels = df['label'].values
    
    if le is None:
        le = LabelEncoder()
        labels = le.fit_transform(labels)
    else:
        labels = le.transform(labels)
    
    labels = to_categorical(labels, num_classes=7)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, le

# Create dataset with label encoder
le = LabelEncoder()
le.fit(train['label'])
train_dataset, _ = create_dataset(train, le=le, batch_size=32)
test_dataset, _ = create_dataset(test, le=le, batch_size=32, shuffle=False)

# Create a ResNet-based model for facial emotion recognition
def residual_block(x, filters, strides=1, downsample=False):
    identity = x
    
    # First convolution layer
    y = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.1)(y)
    
    # Second convolution layer
    y = Conv2D(filters, kernel_size=3, strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    
    # Skip connection
    if downsample:
        identity = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(x)
        identity = BatchNormalization()(identity)
    
    # Add the skip connection to the output
    output = Add()([identity, y])
    output = LeakyReLU(alpha=0.1)(output)
    
    return output

def create_custom_resnet(input_shape=(96, 96, 1), num_classes=7):
    inputs = Input(shape=input_shape)
    
    # Initial conv layer
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=2)(x)
    
    # ResNet blocks
    # Block 1
    x = residual_block(x, 64, strides=1, downsample=True)
    x = residual_block(x, 64)
    
    # Block 2
    x = residual_block(x, 128, strides=2, downsample=True)
    x = residual_block(x, 128)
    
    # Block 3
    x = residual_block(x, 256, strides=2, downsample=True)
    x = residual_block(x, 256)
    
    # Block 4
    x = residual_block(x, 512, strides=2, downsample=True)
    x = residual_block(x, 512)
    
    # Global average pooling and dropout
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    # Fully connected layers
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='custom_resnet')
    return model

# Create and compile the model
model = create_custom_resnet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Define callbacks to improve training
callbacks = [
    ModelCheckpoint(
        'best_emotion_model_v2.h5', 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
]

# Train the model with class weights
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save the final model
model.save('emotion_model_final_v2.h5')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history_v2.png')
plt.show()

# Prediction function
def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(96, 96))
    img = np.array(img) / 255.0
    img = img.reshape(1, 96, 96, 1)
    return img

# Test prediction on a few images
def test_prediction(image_path, true_label):
    print(f"Original image is of {true_label}")
    img = preprocess_image(image_path)
    pred = model.predict(img)
    pred_label = le.inverse_transform([pred.argmax()])[0]
    print(f"Model prediction is {pred_label}")
    plt.imshow(img.reshape(96, 96), cmap='gray')
    plt.title(f"True: {true_label}, Predicted: {pred_label}")
    plt.axis('off')
    plt.show()

# Test on a few examples
test_prediction('images/train/happy/7.jpg', 'happy')
test_prediction('images/train/surprise/15.jpg', 'surprise')