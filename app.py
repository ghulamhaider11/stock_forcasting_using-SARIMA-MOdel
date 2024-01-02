# step1: Import main Libraries
import pandas as pd 
import streamlit as st
import numpy as np 
import seaborn as sns
import plotly.graph_objects as go   
import plotly.express as px 
import datetime 
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm 
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
# step2: Making an app 
#... tile 
app_name = 'Stock Market Forecasting'
st.title(app_name)
st.subheader('This app forecasts the stock market price of the selected company')

# take the input from the user of the app about the start and end date

# sidebar
st.sidebar.header('select the parameter from below')
start_date = st.sidebar.date_input('start_date', date(2020, 1, 1))
end_date = st.sidebar.date_input('end_date', date(2020, 12, 31))
# add ticker symbol list 
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "FB", "ADBE", "TSLA", "PYPl", "INTC", "CMSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Please Select The Company ', ticker_list)

# manually set cache key
data_key = f"{start_date}-{end_date}-{ticker}"

# fetch data from the user using yfinance library with caching
@st.cache_data
def load_data(start_date, end_date, ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# fetch data and add "Date" as a column
data = load_data(start_date, end_date, ticker)
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from',start_date,'to',end_date)
st.dataframe(data)

#plot the data 

st.header('Data Visualization')
st.subheader('plot of the data')
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock',width=1000,height=700,template='plotly_dark')
st.plotly_chart(fig)

#add select box to select the column
column = st.selectbox('Select the column for forecasting',data.columns[1:])
#subseeting the data
data = data[['Date', column]]
st.write("selecting  Data")
st.write(data)

#ADF Test check stationary 
adf_result = adfuller(data[column])
st.write("ADF Statistic:", adf_result[0])
st.write("P-value:", adf_result[1])

if adf_result[1] <= 0.05:
    st.success("The data is stationary.")
else:
    st.warning("The data is not stationary.")

#data decomposition


st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column],model='additive ',period=12)
st.write(decomposition.plot())
 
st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend ,title='Trend',width=1200,height=400,labels={'x':'Date','y':'Price'}))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal ,title='Seasonaity',width=1200,height=400,labels={'x':'Date','y':'Price'}))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid ,title='Residuals',width=1200,height=400,labels={'x':'Date','y':'Price'}))

#let's run the model
#user input for three parameters of the  model nd seasonal order

p = st.slider('Select the value of p', 0, 5, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)
seasonal_order = st.slider('Select the value of seasonal p', 0, 24, 12)
model = sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model = model.fit()
st.header("Model Summary")
st.write(model.summary())
st.write("---")

#predict the future value forcasting 
st.write("<p style='color:green;font-size: 50px;font-weight: bold;'>Forecasting the Data</p>", unsafe_allow_html=True)
forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)

# Predict the future values
predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
predictions = predictions.predicted_mean
# st.write("Predictions", predictions)

# Add index to predictions
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, "Date", predictions.index, True)
predictions.reset_index(drop=True, inplace=True)

st.write("Predictions", predictions)
st.write("Actual Data", data)
st.write("---")

#plot the data 
fig = go.Figure()
#add actual data
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='green')))

fig.update_layout(title='Actual vs predicted',xaxis_title='Date',yaxis_title='price',width=1200,height=400)
st.plotly_chart(fig)



# Adding button to toggle between showing and hiding separate plots for actual and predicted values
show_separate_plots = st.button('Show Separate Plots')
hide_plots = st.button('Hide Plots')

if show_separate_plots and not hide_plots:
    # Plot separate graphs for Actual and Predicted
    fig_actual = go.Figure()
    fig_actual.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig_actual.update_layout(title='Actual Plot', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
    st.plotly_chart(fig_actual)

    fig_predicted = go.Figure()
    fig_predicted.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='green')))
    fig_predicted.update_layout(title='Predicted Plot', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
    st.plotly_chart(fig_predicted)

elif hide_plots and not show_separate_plots:
    # Hide separate plots
    st.write("Plots are hidden. Click the button to show.")
else:
    # Default behavior (both buttons not clicked or both clicked)
    st.write("Click the buttons to show or hide separate plots.")



st.write("Thanks for using the app!")
st.write("Author: Ghulam Haider")





    






    

