import yfinance as yf
from datetime import datetime
import streamlit as st # type: ignore
import yfinance as yf
import pandas as pd 
import plotly.graph_objects as go

# Model Initialization and Training
from sklearn.ensemble import RandomForestClassifier
import joblib

# Get the historical Data for the particular stock 
def get_stock_data(stock_name, start_date, end_date):
    region = "NS"
    
    stock_name = stock_name
    start_date = start_date
    end_date   = end_date
     
    stock_ticker = stock_name.upper()+"."+region.upper()
    # start_date = datetime.strptime(start_date, '%d-%m-%Y')
    start_date = start_date.strftime('%Y-%m-%d')
    # end_date = datetime.strptime(end_date, '%d-%m-%Y')
    end_date = end_date.strftime('%Y-%m-%d')
    
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    return  data

    # Data Preparation
def pre_process_data(data):
    data["open/Close"] = data['Open'] / data['Close']
    data["Volume/High"] = data['Volume'] / data['High']
    data['Volume/Low'] = data['Volume'] / data['Low']
    return data

with st.form( key="Initial Data", border=True):
    st.header("Stock Analytics")
    stock_name = st.selectbox("Please Select Stock Name",
                          ("SUZLON", "TATAMOTORS", "ONGC"))
    
    today = datetime.now()
    date_range = st.date_input(
        "Select Date Range",
        (today, today),
        format="DD.MM.YYYY",
    )
    submit = st.form_submit_button("Fetch Data",use_container_width=True)

if(submit == True):    
    with st.container(border = True):
        st.subheader(stock_name)
        data = get_stock_data(stock_name, date_range[0], date_range[1])
        date = data.index
        
        data['Tomorrow']  = data['Close'].shift(-1)
        data = data.dropna()
        data['Price Change'] = round((data.loc[:,'Close'] - data.loc[:,'Tomorrow'])*100/data['Close'],2)
        display_header = data.columns.tolist()
        display_header.remove("Tomorrow")
        
        st.subheader("Historic Data")
        st.dataframe(data.loc[:,display_header],use_container_width=True)
        
        st.subheader("Prices Variation")
        st.line_chart(data.loc[:,['Close','Open']])
                
        fig = go.Figure(data=[go.Candlestick(x=date,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

        st.plotly_chart(fig)
        
        processed_data = pre_process_data(data)
        predictors = processed_data.columns.tolist()
        predictors.remove("Tomorrow")
        predictors.remove("Price Change")
        
        model = RandomForestClassifier(n_estimators=len(predictors), min_samples_split=len(predictors), random_state=1)

        model = joblib.load(r"C:\Users\Anand\Desktop\Coding\projects\stock_market_prediction\models\RandomForrest.pkl")
        last_date_data = processed_data.iloc[-1]
        last_date_data = [last_date_data.loc[predictors].values]
        predicted_value = model.predict(last_date_data)

        st.subheader("Tommorow Prediction Random Forrest ")
        if(predicted_value[0]==0):
            st.info("Price Goes Down",  icon=":material/thumb_down:")      
        else:
            st.info("Price Goes Up",  icon=":material/thumb_up:")