import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database setup
DATABASE_URL = "sqlite:///./financial_advisor.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    symbol = Column(String)
    quantity = Column(Float)
    purchase_price = Column(Float)
    purchase_date = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Authentication functions
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
        return False
    user = User(username=username, password=password)
    db.add(user)
    db.commit()
    db.close()
    return True

# Stock data functions
@st.cache_data(ttl=3600)
def get_stock_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return dict(zip(table['Security'], table['Symbol']))

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

def predict_stock_price(data, days=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    prediction_days = 60
    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
    
    test_data = scaled_data[-prediction_days:]
    x_test = []
    for _ in range(days):
        x_test.append(test_data[len(test_data)-prediction_days:])
        prediction = model.predict(np.array(x_test).reshape(1, prediction_days, 1))
        test_data = np.append(test_data, prediction)
        x_test = []
    
    predictions = scaler.inverse_transform(test_data[prediction_days:].reshape(-1, 1))
    return predictions

def get_recommendation(symbol, data):
    last_price = data['Close'].iloc[-1]
    avg_price = data['Close'].mean()
    momentum = data['Close'].pct_change().mean()
    
    if last_price < avg_price and momentum > 0:
        return "Buy"
    elif last_price > avg_price and momentum < 0:
        return "Sell"
    else:
        return "Hold"

# Streamlit app
st.set_page_config(page_title="Financial Advisor App", layout="wide")

st.title("Financial Advisor App")

# Sidebar for user authentication
auth_status = st.sidebar.empty()
if 'user' not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    auth_choice = st.sidebar.selectbox("Choose action", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if auth_choice == "Login":
        if st.sidebar.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.user = user
                auth_status.success(f"Logged in as {username}")
            else:
                auth_status.error("Invalid username or password")
    else:
        if st.sidebar.button("Register"):
            if register_user(username, password):
                auth_status.success("Registration successful. Please log in.")
            else:
                auth_status.error("Username already exists")
else:
    auth_status.success(f"Logged in as {st.session_state.user.username}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        auth_status.info("Logged out successfully")

if st.session_state.user:
    # Main app content
    stock_list = get_stock_list()
    selected_stock = st.selectbox("Select a stock", list(stock_list.keys()))
    symbol = stock_list[selected_stock]
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End date", datetime.now())
    
    data = fetch_stock_data(symbol, start_date, end_date)
    
    if not data.empty:
        st.subheader(f"{selected_stock} ({symbol}) Stock Price")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Stock Price'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction
        st.subheader("Price Prediction (Next 30 days)")
        predictions = predict_stock_price(data)
        pred_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30)
        pred_df = pd.DataFrame({'Date': pred_dates, 'Predicted Price': predictions.flatten()})
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical'))
        fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Price'], name='Predicted'))
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Recommendation
        recommendation = get_recommendation(symbol, data)
        st.subheader("Recommendation")
        st.write(f"Based on current analysis, the recommendation for {selected_stock} is: **{recommendation}**")
        
        # Portfolio management
        st.subheader("Portfolio Management")
        db = SessionLocal()
        portfolio = db.query(Portfolio).filter(Portfolio.user_id == st.session_state.user.id).all()
        db.close()
        
        if portfolio:
            portfolio_df = pd.DataFrame([(p.symbol, p.quantity, p.purchase_price, p.purchase_date) for p in portfolio],
                                        columns=['Symbol', 'Quantity', 'Purchase Price', 'Purchase Date'])
            portfolio_df['Current Price'] = portfolio_df['Symbol'].apply(lambda x: yf.Ticker(x).history(period="1d")['Close'].iloc[-1])
            portfolio_df['Current Value'] = portfolio_df['Quantity'] * portfolio_df['Current Price']
            portfolio_df['Profit/Loss'] = portfolio_df['Current Value'] - (portfolio_df['Quantity'] * portfolio_df['Purchase Price'])
            portfolio_df['Profit/Loss %'] = (portfolio_df['Profit/Loss'] / (portfolio_df['Quantity'] * portfolio_df['Purchase Price'])) * 100
            
            st.write(portfolio_df)
            
            total_value = portfolio_df['Current Value'].sum()
            total_profit_loss = portfolio_df['Profit/Loss'].sum()
            st.write(f"Total Portfolio Value: ${total_value:.2f}")
            st.write(f"Total Profit/Loss: ${total_profit_loss:.2f}")
            
            fig_portfolio = go.Figure(data=[go.Pie(labels=portfolio_df['Symbol'], values=portfolio_df['Current Value'])])
            st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Add new stock to portfolio
        st.subheader("Add Stock to Portfolio")
        new_stock = st.selectbox("Select stock to add", list(stock_list.keys()), key="new_stock")
        new_symbol = stock_list[new_stock]
        quantity = st.number_input("Quantity", min_value=0.01, step=0.01)
        purchase_price = st.number_input("Purchase Price", min_value=0.01, step=0.01)
        purchase_date = st.date_input("Purchase Date")
        
        if st.button("Add to Portfolio"):
            db = SessionLocal()
            new_portfolio_item = Portfolio(user_id=st.session_state.user.id, symbol=new_symbol, quantity=quantity,
                                           purchase_price=purchase_price, purchase_date=purchase_date)
            db.add(new_portfolio_item)
            db.commit()
            db.close()
            st.success(f"Added {quantity} shares of {new_stock} to your portfolio")
            st.experimental_rerun()

else:
    st.warning("Please log in to access the Financial Advisor App")
