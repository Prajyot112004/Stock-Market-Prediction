import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import date  # Explicitly import date for today()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Streamlit app title
st.title("Stock Market Prediction App")

# Dropdown for company selection
companies = {
    "Apple": "AAPL",
    "Google": "GOOG",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    # Add more companies as needed
}
selected_company = st.selectbox("Select Company", list(companies.keys()))
ticker = companies[selected_company]

# Predict button
if st.button("Predict"):
    with st.spinner("Fetching data and running predictions..."):
        # Your Tiingo API key
        api_key = "<ADD YOUR API KEY>"

        # Endpoint for daily prices
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"

        end_date = date.today().strftime("%Y-%m-%d")

        params = {
            "token": api_key,
            "startDate": "2022-12-31",
            "endDate": end_date
        }

        response = requests.get(url, params=params)
        data = response.json()
        df1 = pd.DataFrame(data)

        # Alpha Vantage News + Sentiment
        AV_API_KEY = '<ADD YOUR API KEY>'  # Replace with your free key

        def fetch_news_sentiment(ticker, time_from, time_to, limit=1000):
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&time_to={time_to}&limit={limit}&apikey={AV_API_KEY}'
            response = requests.get(url)
            data = response.json()
            if 'feed' in data:
                articles = data['feed']
                df = pd.DataFrame(articles)
                df['time_published'] = pd.to_datetime(df['time_published'])
                return df[['title', 'summary', 'time_published', 'overall_sentiment_score']]
            else:
                st.warning(f"Error fetching news: {data.get('Note', 'Check API key/limits')}")
                return pd.DataFrame()

        # Fetch news (split by year to avoid limits)
        all_news = []
        years = range(2020, 2026)
        for year in years:
            time_from = f'{year}0101T0000'
            time_to = f'{year}1231T2359'
            yearly_df = fetch_news_sentiment(ticker, time_from, time_to)
            all_news.append(yearly_df)
        news_df = pd.concat(all_news, ignore_index=True).drop_duplicates(subset=['title'])

        # Aggregate daily sentiment (mean per date)
        news_daily = news_df.groupby(news_df['time_published'].dt.date).agg({'overall_sentiment_score': 'mean'}).reset_index()
        news_daily.columns = ['date', 'sentiment']  # Rename for consistency

        # Merge with stock data on date
        df_close = df1[['date', 'close']].copy()
        df_close['date'] = pd.to_datetime(df_close['date']).dt.date  # Convert to date
        df_close['SMA_10'] = df_close['close'].rolling(window=10).mean()
        df_close['EMA_20'] = df_close['close'].ewm(span=20, adjust=False).mean()
        window_length = 14
        delta_close = df_close['close'].diff()
        gain_close = delta_close.where(delta_close > 0, 0)
        loss_close = -delta_close.where(delta_close < 0, 0)
        avg_gain_close = gain_close.rolling(window=window_length).mean()
        avg_loss_close = loss_close.rolling(window=window_length).mean()
        rs_close = avg_gain_close / avg_loss_close
        df_close['RSI_14'] = 100 - (100 / (1 + rs_close))
        df_close.dropna(inplace=True)
        news_daily['date'] = pd.to_datetime(news_daily['date']).dt.date
        combined_df_close = df_close.merge(news_daily, on='date', how='left')
        combined_df_close['sentiment'] = combined_df_close['sentiment'].fillna(0)
        combined_df_close.fillna(method='ffill', inplace=True)

        df_open = df1[['date', 'open']].copy()
        df_open['date'] = pd.to_datetime(df_open['date']).dt.date
        window_length = 14
        delta_open = df_open['open'].diff()
        gain_open = delta_open.where(delta_open > 0, 0)
        loss_open = -delta_open.where(delta_open < 0, 0)
        avg_gain_open = gain_open.rolling(window=window_length).mean()
        avg_loss_open = loss_open.rolling(window=window_length).mean()
        rs_close = avg_gain_open / avg_loss_open
        df_open.dropna(inplace=True)
        combined_df_open = df_open.merge(news_daily, on='date', how='left')
        combined_df_open['sentiment'] = combined_df_open['sentiment'].fillna(0)
        combined_df_open.fillna(method='ffill', inplace=True)

        # Feature Engineering
        def prepare_features(df, target_col):
            df_features = df[[target_col, 'sentiment']].copy()
            for i in range(1, 61):
                df_features[f'{target_col}_lag_{i}'] = df_features[target_col].shift(i)
            df_features = df_features.dropna()
            return df_features

        df_features_open = prepare_features(combined_df_open, 'open')
        df_features_close = prepare_features(combined_df_close, 'close')

        # Scale and create sequences
        scaler = MinMaxScaler()
        scaled_data_open = scaler.fit_transform(df_features_open)
        scaled_data_close = scaler.fit_transform(df_features_close)  

        def create_sequences(data, window=60):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i-window:i])
                y.append(data[i, 0])  # Predict target (open/close)
            return np.array(X), np.array(y)

        X_open, y_open = create_sequences(scaled_data_open)
        X_close, y_close = create_sequences(scaled_data_close)

        X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(X_open, y_open, test_size=0.2, random_state=42)
        X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, test_size=0.2, random_state=42)

        # Build and Train LSTM Models
        def build_lstm(input_shape):
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape, activation='relu'))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        model_open = build_lstm((X_train_open.shape[1], X_train_open.shape[2]))
        model_open.fit(X_train_open, y_train_open, epochs=50, batch_size=32, validation_data=(X_test_open, y_test_open), verbose=0)

        model_close = build_lstm((X_train_close.shape[1], X_train_close.shape[2]))
        model_close.fit(X_train_close, y_train_close, epochs=50, batch_size=32, validation_data=(X_test_close, y_test_close), verbose=0)

        # Evaluate MSE
        y_test_open_predict = model_open.predict(X_test_open)
        y_test_close_predict = model_close.predict(X_test_close)
        mse_open = mean_squared_error(y_test_open, y_test_open_predict)
        mse_close = mean_squared_error(y_test_close, y_test_close_predict)
        st.write(f"MSE for Open Price: {mse_open:.4f}")
        st.write(f"MSE for Close Price: {mse_close:.4f}")

        # Plot 1: Monthly Actual vs Predicted Close
        y_train_actual = y_train_close
        y_test_actual = y_test_close
        y_train_predict = model_close.predict(X_train_close)
        y_test_predict = model_close.predict(X_test_close)

        y_train_actual_reshaped = np.column_stack((y_train_actual, np.zeros((len(y_train_actual), X_train_close.shape[2] - 1))))
        y_test_actual_reshaped = np.column_stack((y_test_actual, np.zeros((len(y_test_actual), X_test_close.shape[2] - 1))))
        y_train_predict_reshaped = np.column_stack((y_train_predict.flatten(), np.zeros((len(y_train_predict), X_train_close.shape[2] - 1))))
        y_test_predict_reshaped = np.column_stack((y_test_predict.flatten(), np.zeros((len(y_test_predict), X_test_close.shape[2] - 1))))

        y_train_actual_original = scaler.inverse_transform(y_train_actual_reshaped)[:, 0]
        y_test_actual_original = scaler.inverse_transform(y_test_actual_reshaped)[:, 0]
        y_train_predict_original = scaler.inverse_transform(y_train_predict_reshaped)[:, 0]
        y_test_predict_original = scaler.inverse_transform(y_test_predict_reshaped)[:, 0]

        train_dates = pd.to_datetime(combined_df_close['date'].iloc[60:60 + len(y_train_close)])
        test_dates = pd.to_datetime(combined_df_close['date'].iloc[60 + len(y_train_close):60 + len(y_train_close) + len(y_test_close)])
        train_df = pd.DataFrame({'date': train_dates, 'actual': y_train_actual_original, 'predict': y_train_predict_original})
        test_df = pd.DataFrame({'date': test_dates, 'actual': y_test_actual_original, 'predict': y_test_predict_original})
        full_df = pd.concat([train_df, test_df])

        if not pd.api.types.is_datetime64_any_dtype(full_df['date']):
            full_df['date'] = pd.to_datetime(full_df['date'])

        monthly_data = full_df.groupby(full_df['date'].dt.to_period('M')).mean()[['actual', 'predict']]
        months = monthly_data.index.astype(str)
        x_positions = range(len(months))

        st.header("Actual vs Predicted Closing Prices")
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(x_positions, monthly_data['actual'], label='Actual Monthly Close', color='blue', marker='o')
        ax1.plot(x_positions, monthly_data['predict'], label='Predicted Monthly Close', color='orange', linestyle='--', marker='o')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(months, rotation=45)
        ax1.set_title(f'{selected_company} ({ticker}) Stock Price: Actual vs Predicted (Monthly Average, {train_dates.min().year}-{test_dates.max().year})')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Closing Price ($)')
        ax1.legend()
        ax1.grid(True)
        fig1.tight_layout()
        st.pyplot(fig1)

        # Plot 2: Historical Close vs Predicted Next 30 Days
        last_sequence = X_test_close[-1]
        future_predictions = []
        current_sequence = last_sequence.copy()
        for _ in range(30):
            next_pred = model_close.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
            future_predictions.append(next_pred[0, 0])
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = next_pred[0, 0]

        future_predictions_reshaped = np.column_stack((future_predictions, np.zeros((len(future_predictions), X_test_close.shape[2] - 1))))
        future_predictions_original = scaler.inverse_transform(future_predictions_reshaped)[:, 0]

        last_date = pd.to_datetime(combined_df_close['date'].iloc[-1]).replace(tzinfo=None)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        last_60_dates = pd.to_datetime(combined_df_close['date'].iloc[-60:])
        last_60_closes = combined_df_close['close'].iloc[-60:]

        print("Last historical date:", last_60_dates.iloc[-1])
        print("First predicted date:", future_dates[0])

        last_60_dates_series = pd.Series(last_60_dates)
        future_dates_series = pd.Series(future_dates)
        all_dates = pd.concat([last_60_dates_series, future_dates_series]).reset_index(drop=True)
        all_values = np.concatenate([last_60_closes, future_predictions_original])

        st.header("Closing Price Prediction of next 30 days")
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        ax2.plot(all_dates, all_values, label='Historical Close', color='blue')
        ax2.plot(all_dates.iloc[len(last_60_closes):], future_predictions_original, label='Predicted Next 30 Days', color='orange', linestyle='--', marker='o')
        ax2.axvline(x=all_dates.iloc[len(last_60_closes) - 1], color='red', linestyle=':', label='Prediction Start')
        ax2.set_title(f'{selected_company} ({ticker}) Stock Price Prediction (Next 30 Days from {last_date.date()})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Closing Price ($)')
        ax2.legend()
        ax2.tick_params(rotation=45)
        ax2.grid(True)
        fig2.tight_layout()
        st.pyplot(fig2)

        # Plot 3: Monthly Actual vs Predicted Open
        y_train_actual = y_train_open
        y_test_actual = y_test_open
        y_train_predict = model_open.predict(X_train_open)
        y_test_predict = model_open.predict(X_test_open)

        y_train_actual_reshaped = np.column_stack((y_train_actual, np.zeros((len(y_train_actual), X_train_open.shape[2] - 1))))
        y_test_actual_reshaped = np.column_stack((y_test_actual, np.zeros((len(y_test_actual), X_train_open.shape[2] - 1))))
        y_train_predict_reshaped = np.column_stack((y_train_predict.flatten(), np.zeros((len(y_train_predict), X_train_open.shape[2] - 1))))
        y_test_predict_reshaped = np.column_stack((y_test_predict.flatten(), np.zeros((len(y_test_predict), X_test_open.shape[2] - 1))))

        y_train_actual_original = scaler.inverse_transform(y_train_actual_reshaped)[:, 0]
        y_test_actual_original = scaler.inverse_transform(y_test_actual_reshaped)[:, 0]
        y_train_predict_original = scaler.inverse_transform(y_train_predict_reshaped)[:, 0]
        y_test_predict_original = scaler.inverse_transform(y_test_predict_reshaped)[:, 0]

        train_dates = pd.to_datetime(combined_df_open['date'].iloc[60:60 + len(y_train_open)])
        test_dates = pd.to_datetime(combined_df_open['date'].iloc[60 + len(y_train_open):60 + len(y_train_open) + len(y_test_open)])
        train_df = pd.DataFrame({'date': train_dates, 'actual': y_train_actual_original, 'predict': y_train_predict_original})
        test_df = pd.DataFrame({'date': test_dates, 'actual': y_test_actual_original, 'predict': y_test_predict_original})
        full_df = pd.concat([train_df, test_df])

        if not pd.api.types.is_datetime64_any_dtype(full_df['date']):
            full_df['date'] = pd.to_datetime(full_df['date'])

        monthly_data = full_df.groupby(full_df['date'].dt.to_period('M')).mean()[['actual', 'predict']]
        months = monthly_data.index.astype(str)
        x_positions = range(len(months))

        st.header("Actual vs Predicted Opening Prices")
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        ax3.plot(x_positions, monthly_data['actual'], label='Actual Monthly Open', color='blue', marker='o')
        ax3.plot(x_positions, monthly_data['predict'], label='Predicted Monthly Open', color='orange', linestyle='--', marker='o')
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(months, rotation=45)
        ax3.set_title(f'{selected_company} ({ticker}) Stock Price: Actual vs Predicted (Monthly Average, {train_dates.min().year}-{test_dates.max().year})')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Opening Price ($)')
        ax3.legend()
        ax3.grid(True)
        fig3.tight_layout()
        st.pyplot(fig3)

        # Plot 4: Historical Open vs Predicted Next 30 Days
        last_sequence = X_test_open[-1]
        future_predictions = []
        current_sequence = last_sequence.copy()
        for _ in range(30):
            next_pred = model_open.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
            future_predictions.append(next_pred[0, 0])
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = next_pred[0, 0]

        future_predictions_reshaped = np.column_stack((future_predictions, np.zeros((len(future_predictions), X_test_open.shape[2] - 1))))
        future_predictions_original = scaler.inverse_transform(future_predictions_reshaped)[:, 0]

        last_date = pd.to_datetime(combined_df_open['date'].iloc[-1]).replace(tzinfo=None)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        last_60_dates = pd.to_datetime(combined_df_open['date'].iloc[-60:])
        last_60_opens = combined_df_open['open'].iloc[-60:]

        print("Last historical date:", last_60_dates.iloc[-1])
        print("First predicted date:", future_dates[0])

        last_60_dates_series = pd.Series(last_60_dates)
        future_dates_series = pd.Series(future_dates)
        all_dates = pd.concat([last_60_dates_series, future_dates_series]).reset_index(drop=True)
        all_values = np.concatenate([last_60_opens, future_predictions_original])

        st.header("Opening Price Prediction for next 30 days")
        fig4, ax4 = plt.subplots(figsize=(14, 7))
        ax4.plot(all_dates, all_values, label='Historical Open', color='blue')
        ax4.plot(all_dates.iloc[len(last_60_opens):], future_predictions_original, label='Predicted Next 30 Days (Open)', color='orange', linestyle='--', marker='o')
        ax4.axvline(x=all_dates.iloc[len(last_60_opens) - 1], color='red', linestyle=':', label='Prediction Start')
        ax4.set_title(f'{selected_company} ({ticker}) Stock Price (Open) Prediction (Next 30 Days from {last_date.date()})')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Opening Price ($)')
        ax4.legend()
        ax4.tick_params(rotation=45)
        ax4.grid(True)
        fig4.tight_layout()
        st.pyplot(fig4)

        # Plot 5: Predicted Open vs Close for Next 30 Days
        last_sequence_open = X_test_open[-1]
        future_predictions_open = []
        current_sequence_open = last_sequence_open.copy()
        for _ in range(30):
            next_pred_open = model_open.predict(current_sequence_open.reshape(1, current_sequence_open.shape[0], current_sequence_open.shape[1]))
            future_predictions_open.append(next_pred_open[0, 0])
            current_sequence_open = np.roll(current_sequence_open, -1, axis=0)
            current_sequence_open[-1, 0] = next_pred_open[0, 0]

        future_predictions_open_reshaped = np.column_stack((future_predictions_open, np.zeros((len(future_predictions_open), X_test_open.shape[2] - 1))))
        future_predictions_open_original = scaler.inverse_transform(future_predictions_open_reshaped)[:, 0]

        last_sequence_close = X_test_close[-1]
        future_predictions_close = []
        current_sequence_close = last_sequence_close.copy()
        for _ in range(30):
            next_pred_close = model_close.predict(current_sequence_close.reshape(1, current_sequence_close.shape[0], current_sequence_close.shape[1]))
            future_predictions_close.append(next_pred_close[0, 0])
            current_sequence_close = np.roll(current_sequence_close, -1, axis=0)
            current_sequence_close[-1, 0] = next_pred_close[0, 0]

        future_predictions_close_reshaped = np.column_stack((future_predictions_close, np.zeros((len(future_predictions_close), X_test_close.shape[2] - 1))))
        future_predictions_close_original = scaler.inverse_transform(future_predictions_close_reshaped)[:, 0]

        last_date = pd.to_datetime(combined_df_open['date'].iloc[-1]).replace(tzinfo=None)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')


        st.header("Opening and Closing Price Prediction for next 30 days")
        fig5, ax5 = plt.subplots(figsize=(14, 7))
        ax5.plot(future_dates, future_predictions_open_original, label='Predicted Open Price', color='blue', marker='o')
        ax5.plot(future_dates, future_predictions_close_original, label='Predicted Close Price', color='orange', marker='o', linestyle='--')
        ax5.set_title(f'{selected_company} ({ticker}) Stock Price Prediction (Next 30 Days from {last_date.date()})')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Predicted Stock Price ($)')
        ax5.legend()
        ax5.tick_params(rotation=45)
        ax5.grid(True)
        fig5.tight_layout()

        st.pyplot(fig5)
