import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([[0],[1],[1],[0]])
model = Sequential()
model.add(Dense(2, activation='relu', input_shape=(2,)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(inputs, outputs, epochs=1000, verbose=0)
predictions = model.predict(inputs)
print("Predictions for XOR problem:")
for i in range(len(inputs)):
    print(f"Input: {inputs[i]} -> Predicted Output: {predictions[i][0]:.4f} "
          f"(Threshold: 0.5) -> Class: {1 if predictions[i][0] > 0.5 else 0}")