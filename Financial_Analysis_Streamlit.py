## Package For Project
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
import datetime
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
import plotly.express as px
from sklearn.metrics import precision_score

### Page Configure ###
st.set_page_config(page_title = "Real-Time Data Science Dashboard",
                    page_icon ='✅',
                    layout = 'wide')
st.title("Welcome To My Stock Analysis Project!👨‍💻")


Placeholder      = st.empty()     # All Data Visualization
CandleStickGraph = st.container()
Information      = st.container()
Designer         = st.container()
Sidebar          = st.container()

with Sidebar:
    with st.sidebar:         
        st.sidebar.subheader('Stock Symbol(Ticker) 📥')
        symbol = st.sidebar.text_input('Write your Ticker (ex. AAPL,HPQ,AMZN)','AAPL')
        if symbol is not None:
            textsplit = symbol.replace(' ',',').split(',')
        else:
            textsplit = symbol   

### Page Dashboard ####

with Placeholder.container():

    
    StartDate,EndDate = st.columns(2)
    with StartDate:
        start_date = st.sidebar.date_input('Start Date 🗓', datetime.date(2010, 1, 1))
        st.sidebar.write('Your Stock Starting Date',start_date)
        
        if start_date < datetime.date(2016,1,1):
            st.sidebar.success('System calculating yours data sets')      
        else:
            st.sidebar.error('Please Your Date start before 2016,01,01 for prediction')
    
    with EndDate:
        end_date = st.sidebar.date_input('End Date 🗓',datetime.date(2023,1,1))            
        st.sidebar.write('End Date',end_date)
        date = end_date.year - start_date.year

        if date < 8:
            st.sidebar.error(f'Please your date more than {end_date} for prediction')             
        else:
            st.sidebar.success('System calculating yours data sets') 
    
    symbol_ticker = st.sidebar.selectbox('Select your Stock Symbol 📃',textsplit)
    
    stock = yf.download(symbol_ticker,start_date,end_date)
    stock = stock.reset_index()
    stock['Tomorrow'] = stock['Close'].shift(-1)
    stock['Tomorrow'] = stock['Tomorrow'].fillna(value = 0)
    stock_dec = stock.describe()
    
    Full_name_of_company = ['longName']
    Title_of_company = []
    
    for list in Full_name_of_company:
        title = yf.Ticker(symbol_ticker).info
        title_info = title.get(list)
        Title_of_company.append(title_info)
    st.title(Title_of_company[0])
    
    Graph_1,Graph_2 = st.columns(2)
    
    with Graph_1:
        tab1,tab2= st.tabs([f"📈 StockTime Line Graph {Title_of_company[0]}", "🕵🏽‍♂️ Summary Of Stock "])    
        Line_Graph = px.line(stock, x=stock['Date'],y= stock['Close'])
        Line_Graph.update_layout(title = 'Close Price Time Series Line Graph')
        tab1.plotly_chart(Line_Graph)   
        tab2.dataframe(stock.describe())
    
    ## ML Part 1
    # After separating the dataset, we now have numpy arrays named **X** containing the features, and **y** containing the labels.
    
    X,y = stock['Tomorrow'].values, stock['Close'].values
    
    # Split data 70%-30% into training set and test set
    
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    
    model = DecisionTreeRegressor().fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    statistic_value = pd.DataFrame([{'Mean Square Error': mse,'Root Mean Square Error':rmse,'R Square':r2}],index =['Statistic value'])


    ## ML Part 2
    stock_1 = yf.download(symbol_ticker,start_date,end_date)
    stock_1 = stock_1.reset_index()
    stock_1['Tomorrow'] = stock_1['Close'].shift(-1)
    stock_1['Tomorrow'] = stock_1['Tomorrow'].fillna(value= 0)
    stock_1['Target'] = (stock_1['Tomorrow'] > stock_1['Close']).astype(int)

    horizons = [2,5,60,250,1000]
    predictors = []
    
    for horizon in horizons:
        rolling_averages = stock_1.rolling(horizon).mean()

        ratio_column = f'Close_Ratio_{horizon}'
        stock_1[ratio_column] = stock_1['Close'] / rolling_averages['Close']

        trend_column = f'Trend_{horizon}'
        stock_1[trend_column] = stock_1.shift(1).rolling(horizon).sum()['Target']

        predictors += [ratio_column,trend_column]
    
    stock_1 = stock_1.dropna()
   
    
    model_1 = RandomForestClassifier(n_estimators= 100, min_samples_split= 25, random_state= 0)
    
    train = stock_1.iloc[:-100]
    test = stock_1.iloc[-100:]

    predictors = ["Close", "Volume", "Open", "High", "Low"]
    model_1.fit(train[predictors],train['Target'])
    
    def predict(train, test, predictors, model_1):
        model_1.fit(train[predictors], train["Target"])
        preds = model_1.predict(test[predictors])
        preds[preds >= .6] = 1
        preds[preds < .6 ] = 0 
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined
    
    def backtest(data, model_1, predictors, start=1000, step=50):
        all_predictions = []

        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            predictions_1 = predict(train, test, predictors, model_1)
            all_predictions.append(predictions_1)
            
        return pd.concat(all_predictions)
    
    predictions_1 = backtest(stock_1, model_1, predictors)
    p_score = precision_score(predictions_1['Target'],predictions_1['Predictions'])
    
    statistic_value['Precision Score'] = p_score

    with Graph_2:
        tab3,tab4= st.tabs(['📈 Daily Stock Close Price Predictions', "🕵🏽‍♀️ Statistics Values "])
        Scatter = go.Figure([go.Scatter(x = y_test, y = predictions, name = 'test', mode='markers'),
                        go.Scatter(x=X_train,y=y_train,name = 'test',mode='markers'),
                        go.Scatter(x=y_test,y=p(y_test),name ='prediction')])
        Scatter.update_xaxes(title_text ='Actual Label')
        Scatter.update_yaxes(title_text = 'Predictions')
        Scatter.update_layout(title = 'With Machine Learning Scatter Plot Graph')
        
        tab3.plotly_chart(Scatter)
        

        dataframe2 = tab4.container()       
        
        with dataframe2:
            st.dataframe(statistic_value)
            Fast_info = ['currency','timezone','marketCap','fiftyDayAverage','twoHundredDayAverage']
            Fast_info_column = ['Currency','Time Zone','Market Capitalization','Fifty Day Average', 'Two Hundered Day Average']
            Table_of_Sum = []
            
            ticker = yf.Ticker(symbol_ticker)
            
            for list in Fast_info:
                info = ticker.fast_info.get(list)
                Table_of_Sum.append(info)
            Table_of_Sum = pd.DataFrame(Table_of_Sum, index = Fast_info_column)
            Table_of_Sum = Table_of_Sum.reset_index()
            Table_of_Sum.rename(columns = {0:'Company Information'},inplace = True)
            Table_of_Sum.rename(columns ={'index':'Information'},inplace = True)
            pd.options.display.float_format = '{:,.2f}'.format
            st.dataframe(Table_of_Sum, width = 1200)
    
with CandleStickGraph:
            st.subheader(f'Stock Price Candle Stick Graph {symbol_ticker}')
            CandleStick = go.Figure(data=[go.Candlestick(x = stock['Date'],
                                                     open= stock['Open'],
                                                     high= stock['High'],
                                                     low= stock['Low'],
                                                     close=stock['Close'])])
            CandleStick.update_xaxes(title_text = 'Date')
            CandleStick.update_yaxes(title_text = 'Price')
            CandleStick.update_layout(autosize=False, width = 1200, height = 500)
            
            st.plotly_chart(CandleStick)
            
            st.markdown(f"### Detailed Data View {symbol_ticker}")
            st.dataframe(stock,width = 1200)
            
            metric1,metric2 = st.columns(2)
            
            delta = stock['Open']
            delta_value = delta.iloc[-1]
            delta_value_1 = delta.iloc[-2]

            if delta_value > delta_value_1:
                metric1.metric(label =' Open Price',value = delta_value, delta = delta_value_1, delta_color = "normal")
            else:
                metric1.metric(label =' Open Price',value = delta_value, delta = delta_value_1, delta_color = 'inverse')
            
            delta1 = stock['Close']
            delta1_value = delta1.iloc[-1]
            delta1_value_1 = delta1.iloc[-2] 
            
            if delta1_value > delta1_value_1:
                metric2.metric(label = "Close Price",value = delta1_value, delta= delta1_value_1,delta_color = "normal")
            else:
                metric2.metric(label = "Close Price",value = delta1_value, delta= delta1_value_1,delta_color = "inverse")
        
with Information:
    st.title(f' 🏢 Company Defination {Title_of_company[0]}')
    
    text = []
    website = []
    information = yf.Ticker(symbol_ticker).info
    summary     = information.get('longBusinessSummary')
    text.append(summary)
    
    st.write('Information about Company',text)
    
    get_data = ['longName','website','industry','city','state','country','financialCurrency']
    get_data_index = ['Long Name Of Company','WebSite','Industry','City','State','Country','Financial Currency']
    get_data_list = []
    
    for getData in get_data:
        info = yf.Ticker(symbol_ticker).info
        get  = info.get(getData)
        get_data_list.append(get)
    get_data_frame = pd.DataFrame(get_data_list,index = get_data_index)
    get_data_frame.rename(columns ={0:'Company Information'},inplace = True)
    st.dataframe(get_data_frame, width = 1200)

with Designer:
    st.write('🚀 Preaperd by Batuhan YILDIRIM 🚀')

My_Profile = st.container()

with My_Profile:
    st.text('🥷My Linkedin Account ')
    st.info('https://www.linkedin.com/in/batuhannyildirim/')
    st.text('👻Github Profile')
    st.info('https://github.com/Ybatuhan-EcoBooster')
st.balloons()