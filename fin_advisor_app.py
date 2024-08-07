import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Create a directory for the database if it doesn't exist
os.makedirs("database", exist_ok=True)

# Database setup
DATABASE_URL = "sqlite:///./database/financial_advisor.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

# Portfolio model
class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    symbol = Column(String)
    quantity = Column(Float)
    price = Column(Float)
    current_value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def authenticate_user(username, password):
    db = SessionLocal()
    user = db.query(User).filter(User.username == username, User.password == password).first()
    db.close()
    return user

def register_user(username, password):
    db = SessionLocal()
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        db.close()
        return False  # Username already exists
    user = User(username=username, password=password)
    db.add(user)
    db.commit()
    db.close()
    return True

def load_user_portfolio(user_id):
    db = SessionLocal()
    portfolios = db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
    db.close()
    if portfolios:
        data = pd.DataFrame([{
            'Symbol': portfolio.symbol,
            'Quantity': portfolio.quantity,
            'Price': portfolio.price,
            'Current Value': portfolio.current_value,
            'Timestamp': portfolio.timestamp
        } for portfolio in portfolios])
    else:
        data = pd.DataFrame(columns=['Symbol', 'Quantity', 'Price', 'Current Value', 'Timestamp'])
    return data

def add_transaction(user_id, symbol, quantity, price, current_value):
    db = SessionLocal()
    transaction = Portfolio(user_id=user_id, symbol=symbol, quantity=quantity, price=price, current_value=current_value)
    db.add(transaction)
    db.commit()
    db.close()

# Pre-defined list of popular stocks
popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'BRK-B', 'NVDA', 'JNJ', 'V']

# Fetch top 10 stocks from S&P 500
def get_sp500_top_10():
    sp500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    top_10_symbols = sp500_stocks.head(10)['Symbol'].tolist()
    return top_10_symbols

# Authentication form
auth_mode = st.sidebar.selectbox('Choose Authentication Mode', ['Login', 'Register'])

username = st.sidebar.text_input('Username')
password = st.sidebar.text_input('Password', type='password')

if auth_mode == 'Login':
    if st.sidebar.button('Login'):
        user = authenticate_user(username, password)
        if user:
            st.session_state['user_id'] = user.id
            st.session_state['data'] = load_user_portfolio(user.id)
            st.success('Logged in successfully!')
        else:
            st.error('Invalid username or password')
elif auth_mode == 'Register':
    if st.sidebar.button('Register'):
        if register_user(username, password):
            st.success('Registered successfully! Please log in.')
        else:
            st.error('Username already exists. Please choose a different username.')

if 'user_id' in st.session_state:
    st.success(f"Welcome {username}!")

    # Real-time Stock Price Updates
    def fetch_stock_data(stock_symbol, start_date, end_date):
        return yf.download(stock_symbol, start=start_date, end=end_date, interval='1m')

    st.title("Real-Time Financial Advisor")

    stock_symbol = st.selectbox("Select stock symbol", popular_stocks)
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=1))
    end_date = st.date_input("End date", datetime.now())

    if stock_symbol:
        data = fetch_stock_data(stock_symbol, start_date, end_date)
        st.write(f"Displaying data for {stock_symbol}")
        st.dataframe(data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
        fig.update_layout(title=f'{stock_symbol} Stock Price', xaxis_title='Date', yaxis_title='Price', showlegend=True)
        st.plotly_chart(fig)

        # Machine Learning for Prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data)-time_step-1):
                a = data[i:(i+time_step), 0]
                X.append(a)
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)
        
        time_step = 60
        X_train, Y_train = create_dataset(train_data, time_step)
        X_test, Y_test = create_dataset(test_data, time_step)
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, Y_train, batch_size=1, epochs=1)
        
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index[time_step:len(train_predict)+time_step], y=train_predict.flatten(), mode='lines', name='Train Predict'))
        fig2.add_trace(go.Scatter(x=data.index[len(train_predict)+(time_step*2)+1:len(data)-1], y=test_predict.flatten(), mode='lines', name='Test Predict'))
        fig2.update_layout(title=f'{stock_symbol} Stock Prediction', xaxis_title='Date', yaxis_title='Price', showlegend=True)
        st.plotly_chart(fig2)

        # Portfolio Management with Real-Time Updates
        portfolio = []

        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = pd.DataFrame(columns=['Symbol', 'Quantity', 'Price', 'Current Value', 'Timestamp'])

        def update_portfolio():
            for index, row in st.session_state['portfolio'].iterrows():
                stock_data = fetch_stock_data(row['Symbol'], start_date, end_date)
                if not stock_data.empty:
                    current_price = stock_data['Close'][-1]
                    st.session_state['portfolio'].at[index, 'Current Value'] = current_price * row['Quantity']

        if st.button("Add Transaction"):
            symbol = st.selectbox("Stock Symbol", popular_stocks)
            quantity = st.number_input("Quantity", min_value=0)
            price = st.number_input("Price", min_value=0.0)
            if st.button("Add"):
                current_value = price * quantity
                add_transaction(st.session_state['user_id'], symbol, quantity, price, current_value)
                new_transaction = {'Symbol': symbol, 'Quantity': quantity, 'Price': price, 'Current Value': current_value, 'Timestamp': datetime.utcnow()}
                st.session_state['portfolio'] = st.session_state['portfolio'].append(new_transaction, ignore_index=True)
                st.success(f"Added {quantity} of {symbol} at {price} per unit")

        update_portfolio()

        st.write("Your Portfolio:")
        st.write(st.session_state['portfolio'])

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=st.session_state['portfolio']['Symbol'], y=st.session_state['portfolio']['Current Value'], mode='markers', name='Current Value'))
        fig3.update_layout(title='Portfolio Value', xaxis_title='Stock Symbol', yaxis_title='Current Value', showlegend=True)
        st.plotly_chart(fig3)

        # Display Top 10 S&P 500 Stocks
        st.header("Top 10 S&P 500 Stocks")
        top_10_symbols = get_sp500_top_10()
        top_10_data = yf.download(top_10_symbols, period="1d", interval="1m")
        
        fig4 = go.Figure()
        for symbol in top_10_symbols:
            symbol_data = top_10_data.xs(symbol, level=1, axis=1)
            fig4.add_trace(go.Scatter(x=symbol_data.index, y=symbol_data['Close'], mode='lines', name=symbol))
        
        fig4.update_layout(title='Top 10 S&P 500 Stocks', xaxis_title='Date', yaxis_title='Price', showlegend=True)
        st.plotly_chart(fig4)

        # Rerun every minute
        time.sleep(60)
        st.experimental_rerun()

else:
    st.warning('Please log in to access the application.')
