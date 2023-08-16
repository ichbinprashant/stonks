# stonks
![image](https://github.com/ichbinprashant/stonks/assets/33893505/a293c589-7b91-44b7-b7ff-0582bc365387)
Link to Google Colab: https://colab.research.google.com/drive/1jRkmtz3VF7TxobEgeZ6WdXFSwuKvdTxa?usp=sharing
A stock price predicting algorithm using Long Short-Term Memory (LSTM) architecture of RNN.
This program predicts the closing stock price of a corporation or index using the past 60 day closing stock price.
It works on data collected from Yahoo Finance(https://in.finance.yahoo.com/) using pandas datareader. In this particular case I predicted the price of SBI listed in NSE. The corporation or index needed for prediction can be changed in the 11th line of file stonks.py; using the respective code for a corporation or index in Yahoo Finance.


df = web.DataReader('SBIN.NS', data_source='yahoo', start='2015-01-01', end='2020-11-13')

E.g 'SBIN.NS' is the code for State Bank of India, NSE.  

I've also incorporated matplotlib to visually represent the prediction and the actual price of the market.
