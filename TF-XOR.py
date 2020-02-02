### A quick neural network in TensorFlow/Keras to train the logical XOR-function.
### This is mostly to contrast the MatLab/Octave version written by hand.

# Importing the packages
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Values of the logical XOR.
training_data = np.array([[0.0, 0.0], [0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
training_labels = np.array([0.0, 1.0, 1.0, 0.0])

# We use a topology of a feed-forward network with one hidden layer of
# three neurons.
model = keras.Sequential([
    keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
    keras.layers.Dense(1, activation='linear')
])

# Error is measured in the mead squared, evolution is done via
# Stochastic Gradient Descent, and we track the accuracy of the system.
model.compile(loss = 'mean_squared_error',
              optimizer = 'SGD',
              metrics = ['accuracy'])

# Something like 30 000 epochs seem to suffice, the convergence is very slow for first 10k epochs
EPOCH_NO = 30000

# Train the model.
model.fit(training_data, training_labels, epochs=EPOCH_NO)

# Display the accuracy
predictions = model.predict(training_data)
print('\n\nThe new model says that:')
for j in range(len(predictions)):
    print('({:d} XOR {:d}) = {:1.2f}.'.format(int(training_data[j][0]),
                                              int(training_data[j][1]),
                                              predictions[j][0]))
