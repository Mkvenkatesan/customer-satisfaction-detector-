import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load images from a directory
def load_images_from_dir(directory, label):
    data = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            data.append([img, label])
    return data

# Load images for each emotion label
emotions_dir = {
    "Angry": "D:\vaigaivalley\Projectcentre\dillip\app demo 2\image data set\images\images\train\angry",
    "Disgust": "path_to_disgust_images_directory",
    "Fear": "path_to_fear_images_directory",
    "Happy": "path_to_happy_images_directory",
    "Sad": "path_to_sad_images_directory",
    "Surprise": "path_to_surprise_images_directory",
    "Neutral": "path_to_neutral_images_directory"
}

all_data = []
for emotion, directory in emotions_dir.items():
    print(f"Loading images for {emotion}...")
    data = load_images_from_dir(directory, emotions.index(emotion))
    all_data.extend(data)

# Preprocess data
X = np.array([entry[0] for entry in all_data])
y = np.array([entry[1] for entry in all_data])

# Reshape data for CNN
X = X.reshape(-1, X.shape[1], X.shape[2], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

# Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(emotions), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
batch_size = 32
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=20,
                    validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the trained model
model.save('emotion_detection_model_augmented.h5')
