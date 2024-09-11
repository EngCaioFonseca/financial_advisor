# Real-Time Financial Advisor App
deployed: https://financialadvisorapp.streamlit.app/
## Overview

This Financial Advisor App is a Streamlit-based web application that provides users with real-time stock analysis, price predictions, and portfolio management tools. It's designed to assist users in making informed decisions about their investments by offering stock recommendations and portfolio insights.

![fin_assistance_app](https://github.com/user-attachments/assets/b14852d7-6dfa-4c78-bb9d-83ed3bb00e7b)



## Features

- User Authentication: Secure login and registration system
- Real-time Stock Data: Fetch and display current stock information
- Stock Price Visualization: Interactive candlestick charts for selected stocks
- Price Prediction: 30-day stock price forecast using LSTM neural networks
- Stock Recommendations: Buy/Sell/Hold recommendations based on current market trends
- Portfolio Management: Add stocks to your portfolio and track their performance
- Portfolio Analysis: View total value, profit/loss, and holdings distribution

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/financial-advisor-app.git
   cd financial-advisor-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run fin_advisor_app.py
   ```

2. Open a web browser and go to `http://localhost:8501` (or the URL provided in the terminal).

3. Register for a new account or log in if you already have one.

4. Use the sidebar to navigate through different features of the app.

## Dependencies

- streamlit
- yfinance
- pandas
- plotly
- scikit-learn
- tensorflow
- sqlalchemy

For a complete list of dependencies, see `requirements.txt`.

## Database

The app uses SQLite as its database. The database file (`financial_advisor.db`) will be created in the same directory as the script when you run the app for the first time.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Future Improvements

- Implement more advanced recommendation algorithms
- Add more technical indicators and fundamental analysis
- Incorporate news sentiment analysis for stocks
- Provide more detailed portfolio analytics and rebalancing suggestions
- Implement real-time price updates using websockets

## Disclaimer

This app is for educational purposes only. It should not be considered as financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## License

This project is open source and available under the [MIT License](LICENSE).
