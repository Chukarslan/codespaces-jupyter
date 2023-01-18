import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# Get historical stock data for Microsoft
msft = yf.Ticker("MSFT")
df = msft.history(start='2016-09-01', end='2023-01-05')

# Create a new DataFrame with only the Close column and Friday's data
df = df[['Close']][df.index.weekday == 4]

# Scale the data
scaler = MinMaxScaler()
df[['Close']] = scaler.fit_transform(df[['Close']])

# Create a function to create a dataset with a lookback window
def create_dataset(df, lookback=1):
    dataX, dataY = [], []
    for i in range(len(df)-lookback-1):
        a = df[i:(i+lookback), 0]
        dataX.append(a)
        dataY.append(df[i + lookback, 0])
    return np.array(dataX), np.array(dataY)

# Create the dataset with a lookback of 7 days
lookback = 7
X, Y = create_dataset(df.values, lookback)

# Reshape the data for the 2D LSTM model
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the custom 2D LSTM model
inputs = Input(shape=(X.shape[1], X.shape[2]))
lstm1 = LSTM(64, return_sequences=True)(inputs)
lstm2 = LSTM(32)(lstm1)
dense = Dense(1)(lstm2)
model = Model(inputs=inputs, outputs=dense)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
model.fit(X, Y, epochs=100, batch_size=32, verbose=1)

# Use the model to predict the next day's expected returns
next_day_prediction = model.predict(X[-1].reshape(1, lookback, 1))

# Reverse the scaling
next_day_prediction = scaler.inverse_transform(next_day_prediction)
print("Expected returns for next Friday:", next_day_prediction[0][0])