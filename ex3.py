"""
Author: Scriptone
Created: 2024/12/6
Description: Deep learning example 3: Convolutional Neural Network (CNN) for image classification
"""

# You'll notice some Dutch words in the plots/logs, this is because I'm Dutch and I used this for a presentation to explain to my fellow students how CNNs work.
# I added some debugging plots to include in my presentation, but they are not necessary for the code to work.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset (very popular dataset for image classification)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1, because neural networks work better with small input values, more efficient
train_images = train_images / 255
test_images = test_images / 255

# CIFAR-10 contains 10 categories, each represented by a number from 0 to 9
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Plot the first 25 images from the training set and display the class name below each image (debugging)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Plot the distribution of samples per class in CIFAR-10
plt.figure(figsize=(12, 6))

# Count the number of samples per class
train_counts = np.bincount(
    train_labels.flatten()
)  # Flatten to 1D because cifar labels are 2D
test_counts = np.bincount(test_labels.flatten())

# Make bar plot
x = np.arange(len(class_names))
width = 0.35
plt.bar(x - width / 2, train_counts, width, label="Training set", color="skyblue")
plt.bar(x + width / 2, test_counts, width, label="Test set", color="lightgreen")

# Labels and title.
plt.xlabel("Klasse")
plt.ylabel("Aantal samples")
plt.title("Distributie van samples per klasse in CIFAR-10")
plt.xticks(x, class_names, rotation=45, ha="right")
plt.legend()

# Put the counts on top of the bars
for i, count in enumerate(train_counts):
    plt.text(i - width / 2, count, str(count), ha="center", va="bottom")
for i, count in enumerate(test_counts):
    plt.text(i + width / 2, count, str(count), ha="center", va="bottom")

plt.tight_layout()
# Purpose of this plot is again for debugging.
plt.show()

# Print the number of samples per class in CIFAR-10
print("\nAantal samples per klasse:")
for i, name in enumerate(class_names):
    print(f"{name:12} - Training: {train_counts[i]:5d}, Test: {test_counts[i]:5d}")

# Create the CNN model
model = models.Sequential()

# We're using BatchNormalization and Dropout to prevent overfitting. Dropout randomly sets a fraction of input units to 0 at each update during training time, which helps prevent overfitting.
# The chosen values for Dropout are based on experience (how little that may be) and experimentation, they are not set in stone.
# First conv2D + pooling layer (32 filters, 3x3 kernel, ReLU activation, input shape 32x32x3)
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Second conv2D + pooling layer (64 filters, 3x3 kernel, ReLU activation)
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Third conv2D + pooling layer (128 filters, 3x3 kernel, ReLU activation)
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Flatten layer to convert 3D output to 1D because Dense layers require 1D input
model.add(layers.Flatten())

# Dense layer with 512 neurons and ReLU activation
model.add(layers.Dense(512, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# Output layer with 10 neurons (one for each class) and softmax activation
model.add(layers.Dense(10, activation="softmax"))

# Choose the optimizer: Adam because it's a good general-purpose optimizer
optimizer = tf.keras.optimizers.Adam()

# Choose the loss function: SparseCategoricalCrossentropy because we have integer labels
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# Choosing accuracy because we mostly care about the percentage of correct predictions
metric = ["accuracy"]

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metric)
model.summary()

# Epochs: number of times the model will see the entire dataset
NUM_EPOCHS = 10
# Batch size: number of samples that will be propagated through the network at once (faster training)
BATCH_SIZE = 64

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs=NUM_EPOCHS,
    validation_data=(test_images, test_labels),
    batch_size=BATCH_SIZE,
)

# Plotting the training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validatie Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plotting the loss history
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validatie Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate the model on the test set
predictions = model.predict(test_images)
pred_classes = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.imshow(test_images[i])

    # If the prediction is correct, the title will be green, otherwise red
    if pred_classes[i] == test_labels[i][0]:
        color = "green"  # Correct prediction
    else:
        color = "red"  # Incorrect prediction

    plt.title(
        f"Voorspeld: {class_names[pred_classes[i]]}\nWerkelijk: {class_names[test_labels[i][0]]}",
        color=color,
    )

    # Remove axis because it's not necessary
    plt.axis("off")

plt.tight_layout()
plt.show()

# You probably won't test this so I'll print the output here
"""
Epoch 1/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 14ms/step - accuracy: 0.3207 - loss: 2.2005 - val_accuracy: 0.5277 - val_loss: 1.2915
Epoch 2/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 13ms/step - accuracy: 0.5240 - loss: 1.3358 - val_accuracy: 0.4836 - val_loss: 1.4666
Epoch 3/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 13ms/step - accuracy: 0.5877 - loss: 1.1659 - val_accuracy: 0.6009 - val_loss: 1.1310
Epoch 4/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 15ms/step - accuracy: 0.6206 - loss: 1.0771 - val_accuracy: 0.6654 - val_loss: 0.9472
Epoch 5/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 14ms/step - accuracy: 0.6544 - loss: 0.9892 - val_accuracy: 0.6721 - val_loss: 0.9328
Epoch 6/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 15ms/step - accuracy: 0.6689 - loss: 0.9348 - val_accuracy: 0.4046 - val_loss: 2.1680
Epoch 7/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 16ms/step - accuracy: 0.6867 - loss: 0.8925 - val_accuracy: 0.7119 - val_loss: 0.8195
Epoch 8/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 14ms/step - accuracy: 0.6929 - loss: 0.8635 - val_accuracy: 0.7293 - val_loss: 0.7783
Epoch 9/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 14ms/step - accuracy: 0.7098 - loss: 0.8313 - val_accuracy: 0.6653 - val_loss: 0.9903
Epoch 10/10
[1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 14ms/step - accuracy: 0.7174 - loss: 0.8101 - val_accuracy: 0.7379 - val_loss: 0.7454


"""

# You see that we get a validation accuracy of 73.79% after 10 epochs, which is not bad for a simple CNN model. But there is still room for improvement.
# The goal was merely to show how simple it is to create a CNN model with TensorFlow/Keras and train it on a dataset like CIFAR-10.
# If I get declined I'll probably add some more advanced techniques like Data Augmentation, Early Stopping, Learning Rate Scheduling, etc. to improve the model.

# From the 36 images I plotted, only 8 were wrong which is a 77.78% accuracy on the test set. This is close to the validation accuracy of 73.79%. This is a good sign that the model is not overfitting.
# The model is generalizing well to unseen data, which is the goal of any deep learning model.
