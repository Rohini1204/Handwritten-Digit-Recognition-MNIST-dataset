import idx2numpy 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

x_train = idx2numpy.convert_from_file(
    'dataset/train-images.idx3-ubyte'
)

y_train = idx2numpy.convert_from_file(
    'dataset/train-labels.idx1-ubyte'
)

x_test = idx2numpy.convert_from_file(
    'dataset/t10k-images.idx3-ubyte'
)

y_test = idx2numpy.convert_from_file(
    'dataset/t10k-labels.idx1-ubyte'
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# single image shape
print("\nSingle image shape:")
print(x_train[0].shape)

# raw pixel values
print("\nPixel values:")
print(x_train[0])

# display image 
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Digit Label: {y_train[0]}")
plt.show()

# Normalization
print("\nBefore normalization")
print(x_train[0][0][:30])

x_train = x_train/255.0
x_test = x_test/255.0

print("After Normalization")
print(x_train[0][0][:30])

# Flattening
flattened = x_train[0].flatten()

print("flattened shape")
print(flattened.shape)

print("Flattened image")
print(flattened)

# Building the model
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

print("\n Model created ")

# compiling the model
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
print("Model Compiled")

# Training the model
model.fit(x_train,y_train, epochs=5)

# evaluating
loss,accuracy = model.evaluate(x_test,y_test)
print("Accuracy:", accuracy)

# predict 
prediction = model.predict(
    x_test[:1]
)

print("\nPrediction Probabilities:")
print(prediction)

predicted_digit = np.argmax(prediction)

print("\nPredicted Digit:")
print(predicted_digit)

print("\nActual Digit:")
print(y_test[0])
