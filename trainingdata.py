import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Enable memory growth for TensorFlow (GPU usage)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.VirtualDeviceConfiguration(memory_limit=4096)]  # Set the memory limit (in MB)
    )

# Paths to your datasets
train_dir = r'C:\Users\HP\PycharmProjects\tyredetection\fixed_splittingdata\train'
val_dir = r'C:\Users\HP\PycharmProjects\tyredetection\fixed_splittingdata\val'
test_dir = r'C:\Users\HP\PycharmProjects\tyredetection\fixed_splittingdata\test'

# ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,  # Reduced batch size
    class_mode='binary',
    shuffle=True  # Shuffle the training data for better generalization
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,  # Same batch size for validation
    class_mode='binary',
    shuffle=False  # No need to shuffle validation data
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,  # Same batch size for testing
    class_mode='binary',
    shuffle=False  # No need to shuffle test data
)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"🧾 Test Accuracy: {test_acc:.4f}")

# Save the trained model
model_save_path = r'C:\Users\HP\PycharmProjects\tyredetection\tyre_condition_model.h5'
model.save(model_save_path)

print(f"✅ Model saved at: {model_save_path}")