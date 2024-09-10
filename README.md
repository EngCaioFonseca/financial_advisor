# Real-Time Financial Advisor App

This Streamlit application provides real-time financial advice and portfolio management tools, including stock price tracking, prediction, and portfolio visualization.

## Features

- User authentication (login/register)
- Real-time stock price updates
- Stock price prediction using LSTM
- Portfolio management
- S&P 500 top stocks visualization

## Dependencies

- streamlit
- yfinance
- pandas
- plotly
- sklearn
- keras
- numpy
- sqlalchemy
- matplotlib

## Main Components

1. **Authentication**: Users can register and log in to access their personal portfolio.

2. **Stock Data Fetching**: Real-time stock data is fetched using the yfinance library.

3. **Data Visualization**: 
   - Stock price charts
   - Portfolio value visualization
   - S&P 500 top stocks comparison

4. **Machine Learning Prediction**:
   - Uses LSTM to predict future stock prices
   - Displays train and test predictions

5. **Portfolio Management**:
   - Add transactions to your portfolio
   - Real-time updates of portfolio value

6. **Database Integration**:
   - SQLite database to store user and portfolio information

## Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run fin_advisor_app.py
   ```

3. Register or log in to access the app.

4. Select a stock symbol and date range to view real-time data and predictions.

5. Manage your portfolio by adding transactions.

6. Explore the S&P 500 top stocks visualization.

## Note

This app uses a SQLite database to store user information and portfolio data. Ensure you have write permissions in the directory where the app is running.

## Future Improvements

- Implement more advanced prediction models
- Add more portfolio analysis tools
- Integrate with real trading APIs for live trading capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
