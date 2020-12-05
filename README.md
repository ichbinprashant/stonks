# stonks
A stock price predicting algorithm using Long Short-Term Memory (LSTM) architecture of RNN.
This program predicts the closing stock price of a corporation or index using the past 60 day closing stock price.
It works on data collected from Yahoo Finance(https://in.finance.yahoo.com/) using pandas datareader. In this particular case I predicted the price of SBI listed in NSE. The corporation or index needed for prediction can be changed in the 11th line of file stonks.py; using the respective code for a corporation or index in Yahoo Finance.

df = web.DataReader('SBIN.NS', data_source='yahoo', start='2015-01-01', end='2020-11-13')

E.g 'SBIN.NS' is teh code for State Bank of India, NSE.  
