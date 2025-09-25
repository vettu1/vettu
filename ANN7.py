import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = train_images.reshape(train_images.shape[0], 28, 28)
test_images = test_images.reshape(test_images.shape[0], 28, 28)
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images.reshape(-1, 28 * 28), train_labels,
          epochs=10, batch_size=32)
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28 * 28), test_labels)
print(f'Test accuracy: {test_acc:.4f}')
predictions = model.predict(test_images.reshape(-1, 28 * 28))
def display_predictions(images, labels, predictions):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'True: {labels[i]}\nPredicted: {np.argmax(predictions[i])}')
        plt.axis('off')
    plt.show()
display_predictions(test_images, test_labels, predictions)