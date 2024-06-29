import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st # type: ignore
import yfinance as yf
import pandas as pd 
import plotly.graph_objects as go

# Model Initialization and Training
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sentiment Modules 
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# News Module
from gnews import GNews

# Time series Production using Prophet
from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import plot_plotly # Import the plot_plotly function
from prophet.serialize import model_to_json, model_from_json



def train_prophet_model(stock_name):
    end_date = datetime.today()
    start_date =  end_date - timedelta(days=365)

    data = get_stock_data(stock_name, start_date, end_date)
    close_date = data.loc[:, 'Close']
    date = data.index

    test_data = pd.DataFrame({"ds":date,"y":close_date.values})
    print(test_data)
    m = Prophet()
    m.fit(test_data)
    # Python

    with open(str(stock_name)+'.json', 'w') as fout:
        fout.write(model_to_json(m))  # Save model

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

def get_fundamental_financials(stock_name):

    region = "NS"
    stock_ticker = stock_name.upper()+"."+region.upper()
    fundamental_data = yf.Ticker(stock_ticker).info
    
    return fundamental_data

    # Data Preparation

def pre_process_data(data):
    data["open/Close"] = data['Open'] / data['Close']
    data["Volume/High"] = data['Volume'] / data['High']
    data['Volume/Low'] = data['Volume'] / data['Low']
    return data

def get_sentiment(text):
    scores = [0,0,0]
    MODEL = f"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # model.save_pretrained(MODEL)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    
    scores = list(output[0][0].detach().numpy())
    index = scores.index(max(scores))
    sentiment = ["Negative","Neutral","Positive"]
    return sentiment[index]

def get_news(stock_name):
    google_news = GNews()
    google_news.period = '1d'
    news = google_news.get_news(stock_name)

    news = pd.DataFrame(news)
    news.drop(['publisher', 'published date','description'], axis=1, inplace=True)
    news.rename(columns={'title': 'Headline'}, inplace=True)
       
    return news

with st.sidebar:
    st.header("Stock Analytics")
    stock_name = st.selectbox("Please Select Stock Name",
                          ("SUZLON", "TATAMOTORS", "ONGC"))
    
    today = datetime.now()
    yesterday = today - timedelta(days=7)
    date_range = st.date_input(
        "Select Date Range",
        (yesterday, today),
        format="DD.MM.YYYY",
    )
    
st.title(stock_name)

financials, historic_data, charts, news = st.tabs(["Financials", "Historic Data", "Charts", "News"])
with financials:
    fundamental_financials = get_fundamental_financials(stock_name)
    st.info(fundamental_financials['longBusinessSummary'],icon=":material/info:")
    st.link_button("Website", fundamental_financials['website'])

    # st.json(fundamental_financials)
    EBITDA, D2E, MCap = st.columns(3,gap="small")
    
    with EBITDA:
        st.metric(label="EBITDA",value=fundamental_financials['ebitda'])
    with D2E:
        st.metric(label="Debt To Equity",value=fundamental_financials['debtToEquity'])
    with MCap:
        st.metric(label="Market Cap",value=fundamental_financials['marketCap'])
        
    BV, H52, L52 = st.columns(3,gap="small")
    
    with BV:
        st.metric(label="Book Value",value=fundamental_financials['bookValue'])
    with H52:
        st.metric(label="52 Week High",value=fundamental_financials['fiftyTwoWeekHigh'])
    with L52:
        st.metric(label="52 Week Low",value=fundamental_financials['fiftyTwoWeekLow'])
    
with historic_data:
    
    data = get_stock_data(stock_name, date_range[0], date_range[1])
    date = data.index
    
    data['Tomorrow']  = data['Close'].shift(-1)
    data = data.dropna()
    data['Price Change'] = round((data.loc[:,'Close'] - data.loc[:,'Tomorrow'])*100/data['Close'],2)
    display_header = data.columns.tolist()
    display_header.remove("Tomorrow")
    st.subheader("Historic Data")
    st.dataframe(data.loc[:,display_header],use_container_width=True)

with charts:    

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

    model = joblib.load(r"RandomForrest.pkl")
    last_date_data = processed_data.iloc[-1]
    last_date_data = [last_date_data.loc[predictors].values]
    predicted_value = model.predict(last_date_data)

    st.subheader("Trend Tommorow Prediction By Random Forrest ")
    if(predicted_value[0]==0):
        st.info("Price Goes Down",  icon=":material/thumb_down:")      
    else:
        st.info("Price Goes Up",  icon=":material/thumb_up:")


    st.subheader("Future Trend of the Stock market By Prophet ")
    with open(str(stock_name)+'.json', 'r') as fin:
        m = model_from_json(fin.read())  # Load model
    
    future = m.make_future_dataframe(periods=60)
    forecast = m.predict(future)
    fig = plot_plotly(m, forecast)
    fig.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='white'
    )
    st.plotly_chart(fig)

    
    
with news:
    
    news = get_news(stock_name)
    news['Sentiment'] = news['Headline'].apply(get_sentiment)
    order = ['Headline',"Sentiment", "url"]
    st.subheader("Todays News")
    st.dataframe(news[order])
    avg_sentiment  = news.Sentiment.value_counts()
    avg_sentiment = round((avg_sentiment / avg_sentiment.sum()) * 100,2)
    
    st.subheader("News Sentiment Averages ")
    st.dataframe(avg_sentiment)
    
    

    