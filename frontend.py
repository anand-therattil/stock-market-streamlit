import yfinance as yf
from datetime import datetime
import streamlit as st # type: ignore
import yfinance as yf
import pandas as pd 
import altair as alt

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
        
        st.subheader("Prices Variation")
        st.line_chart(data.loc[:,['Close','Open']])
        
        st.subheader("Historic Data")
        st.dataframe(data.iloc[:,1:],use_container_width=True)
        
        