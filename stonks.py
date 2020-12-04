import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import  LSTM, Dense

plt.style.use('fivethirtyeight')
df = web.DataReader('SBIN.NS', data_source='yahoo', start='2015-01-01', end='2020-11-13')



print(df)





close = df.filter(['Close'])

#close.fillna(method='ffill') #fills the values of the N/A datas with that of the pervious data. e.g. sat sun will have closing price of friday

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 4
plt.rcParams["figure.figsize"] = fig_size
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.grid(True)
plt.plot(close)
plt.show()




dataset = close.values #convers the dataframe into numpy array
#get the no of rows to train the model on
train_data_len = math.ceil(len(dataset)*.8)
print(train_data_len)


#scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


#creating trainging data
#creating scaled training dataset
train_data = scaled_data[0:train_data_len,:]
#splittign the data into x_train and y_train dataset
x_train = []
y_train = [] # this will be the 60th(for this) value which out model will predict

for i in range(60, len(train_data)): #appends from 60 to n, second look will predict the 61st day, 3rd the 62nd day... so each day is trained
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i,0])


#converting x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshaping data (lstm model expects 3D data)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)


#lstm model
model = Sequential()
model.add(LSTM(units= 50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(units= 50, return_sequences=False))
model.add(Dense(units= 25))
model.add(Dense(units= 1))

#compile model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#train the model
model.fit(x_train, y_train, batch_size= 1, epochs= 3)

#creating testing data set
#creating a new array containing remaining data from the data set
test_data = scaled_data[train_data_len -60: , :]

x_test = []
y_test = dataset[train_data_len :, :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])



#converting x_test to numpy arrays
x_test = np.array(x_test)

#reshaping data (lstm model expects 3D data)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test.shape)

#get the model's price predicted values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the root mean square error(RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

#plot the data
train = close[:train_data_len]
valid = close[train_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize = (20,8))
plt.title('Model')
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Closing Price', fontsize= 18)

plt.grid(True)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Training Data', 'Actual Value', 'Prediction'], loc= 'upper left')
plt.show()



print(valid)