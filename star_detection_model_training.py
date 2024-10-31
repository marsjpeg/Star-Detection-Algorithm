import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# CNN Model stuff here (either import a star detection CNN or train my own, what's better?)
# Splitting data up
categories = ['galaxy', 'star']
images = []
labels = []

for category in categories:
    path = os.path.join("Cutout Files", category)
    class_num = categories.index(category) # 0 for galaxy, 1 for star
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))  # Read as grayscale
        img_resized = cv2.resize(img_array, (64, 64))  # Resize to fit model's input size
        images.append(img_resized)
        labels.append(class_num)

images = np.array(images) # .reshape(-1, 64, 64, 3)  # Add channel dimension for grayscale images
print(images.shape)
images = images.astype('float32') / 255.0        # Normalize pixel values to [0, 1]
labels = np.array(labels)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=16, batch_size=32)
model.save('star_verification_cnn_2.h5')


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()


model.evaluate(x_val, y_val)
prediction = model.predict(x_val)
print(prediction)
print(y_val)
