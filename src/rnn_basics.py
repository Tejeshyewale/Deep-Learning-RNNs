import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Dummy dataset
X = np.random.rand(100, 10, 1)
y = np.random.rand(100, 1)

model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(10, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=5)

print("RNN Model Trained Successfully!")
