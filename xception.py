# Import the necessary libraries
import yfinance as yf
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Get the MSFT stock data from yfinance
msft = yf.Ticker("MSFT")
df = msft.history(start='2018-01-01', end='2021-12-31')

# Create a new dataframe with only Fridays
fridays = df[df.index.dayofweek == 4]

# Get the last 7 days of data
lookback = 7
fridays_7d = fridays.tail(lookback)

# Scale the data
scaler = MinMaxScaler()
fridays_7d = scaler.fit_transform(fridays_7d)

# Reshape the data for the Xception model
fridays_7d = np.reshape(fridays_7d, (1, lookback, 4, 1)) if len(fridays_7d)==7 else np.reshape(fridays_7d, (1, len(fridays_7d), 4, 1))

# Create the Xception model
base_model = Xception(weights=None, include_top=False, input_shape=(lookback, 4, 1))
x = base_model.output
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(fridays_7d, epochs=50, batch_size=1, callbacks=[EarlyStopping(monitor='loss', patience=5)])

# Use the model to predict the next day's expected returns
next_day_prediction = model.predict(fridays_7d)

# Reverse the scaling
next_day_prediction = (next_day_prediction * (df[['Close']].max() - df[['Close']].min())) + df[['Close']].min()

# Print the prediction
print(next_day_prediction)