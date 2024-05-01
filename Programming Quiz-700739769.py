#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Time Series Forecasting Task:
#• Load a time series dataset (e.g., stock prices, weather data).
#• Build a recurrent neural network (RNN) or LSTM model using Keras.
#• Train the model to forecast future values based on historical data.
#• Evaluate the model's performance using appropriate metrics (e.g., MAE, RMSE).


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset (example: daily temperature data)
# Replace this with your own dataset loading code
# For demonstration purposes, let's generate a synthetic dataset
data = np.sin(np.arange(1000) * 0.1) + np.random.normal(0, 0.1, 1000)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(data_normalized) * 0.8)
test_size = len(data_normalized) - train_size
train_data, test_data = data_normalized[0:train_size, :], data_normalized[train_size:len(data_normalized), :]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Reshape into X=t and Y=t+1
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
train_mae = mean_absolute_error(y_train[0], train_predict[:,0])
test_mae = mean_absolute_error(y_test[0], test_predict[:,0])

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(np.arange(len(data)), scaler.inverse_transform(data_normalized), label='True Data')
plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, label='Training Predictions')
plt.plot(np.arange(len(train_predict) + 2*time_step, len(data)), test_predict, label='Test Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()



# In[ ]:


#Load the MNIST dataset.
#• Build a simple convolutional neural network (CNN) using Keras Sequential model.
#• Train the CNN model on the MNIST dataset.
#• Evaluate the model's performance on a test set and report accuracy.
#• Use grid search to optimize hyperparameters such as learning rate, batch size, and
#optimizer choice.
#• Use Callback functions to automate training process like “ReduceLROnPlateau” and keep
#check on validation loss. Also use history object for result visualization







import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0

# Build the CNN model
def build_model(learning_rate=0.001):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define callback function
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train the model
model = build_model()
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), callbacks=[reduce_lr])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




