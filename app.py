import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import urllib.error
import pandas_datareader.data as web
from yahooquery import Ticker
import yfinance as yfin
from keras.models import load_model
import streamlit as st


start = dt.date(2011,1,1)
end = dt.date(2020,12,31)

st.title('Stock Market Trend Prediction App')
 
#Sidebar
st.sidebar.subheader('Stock Ticker Parameters')

#Retrieving ticker data
try:
      ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
except urllib.error.URLError as e:
     st.warning(f"Error fetching stock ticker. Kindly check your internet connection and refresh the web page")
     st.stop()

tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list)

try:
      ticker_data = web.get_quote_yahoo(tickerSymbol)
except urllib.error.URLError as e:
     st.warning(f"Error fetching stock ticker data. Kindly check your internet connection and refresh the web page")
     st.stop()
#Retrieving Ticker information

#Ticker Logo
try:
    logo_url = 'https://logo.clearbit.com/{}.com'.format(tickerSymbol)
    st.image(logo_url, width=100)
except:
    st.warning('Logo not found')

#Ticker Name
try:
    string_name = ticker_data.loc[tickerSymbol]['longName']
    st.header('**%s**' % string_name)
except KeyError:
    st.warning('Unable to retrieve company name.')
#Ticker Summary
ticker = Ticker(tickerSymbol)
summary = 'Summary of {} is not available'.format(tickerSymbol)

try:
    summary = ticker.asset_profile[tickerSymbol]['longBusinessSummary']
except KeyError:
    try:
        summary = ticker.asset_profile[tickerSymbol]['longSummary']
    except KeyError:
        try:
            summary = ticker.asset_profile[tickerSymbol]['longDescription']
        except KeyError:
            pass

st.info(summary)


df = yfin.download(tickerSymbol, start, end)

if df.empty:
      st.warning(f"Error fetching stock ticker data. Kindly check your internet connection and refresh the web page")
      st.stop()
                             

#Describing Data
st.subheader('Data from 2011 - 2021')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load my model
model = load_model('keras_model.h5')

#Testing Part

## final_df = past_100_days.append(data_testing, ignore_index=True) -> this will be deprecated soon
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Prediction Graph
st.subheader('Predictions')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Final Graph
st.subheader('Predictions vs Original')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)


#    STOCK MARKET TREND APPLICATION SUMMARY
# A simple text input field where the user inputs the stock ticker. e.g TSLA. User can get it from yahoo!finance.
# The data description which is printed using the df.describe() function - pandas function.
# The data displayed is from 2011 - 2020 divided into training & testing set. 
#       The high, low, open, close, volume & Adj Close for the 9 years
#       With training part being 70% and testing part 30%
# In this project the study focuses on predicting the closing price of a particular day
#        100 days MA- It will take the value of the previous 100 days
#        Sliding average technique using LSTMs
#        Technical analysts have it that (cite) (focus a diagram on 3:09 - 3:15 to explain this)
#                    if 100 days MA crosses above the 200MA 
#                             it is a up-trend 
#                       but if it crosses below the 100MA
#                             it is down-trend        
# The final graph highlghts the original values vs the predicted values
# The bonus of this project is the fact that it implements an already proven technique 'Sliding averages'. 
# That means even if one feels the offset [the gap btwn the original & predicted price] predicted by LSTM is too large. 
# The user can still rely on the 'Sliding averages' using the two displayed graphs

