import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

def fetch_data(ticker):
    df = yf.download(ticker, start='2022-01-01', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df[['Close', 'Volume']].dropna()

def engineer_features(df):
    data = df.copy()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()

    for i in range(1, 6):
        data[f'Lag_{i}'] = data['Close'].shift(i)

    return data.dropna()

def recursive_forecast(model, X_train, y_train, last_features, scaler_y, days):
    model.fit(X_train, y_train.ravel())
    forecasts = []
    curr_feat = last_features.copy().reshape(1, -1)

    for _ in range(days):
        pred_scaled = model.predict(curr_feat)[0]
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        forecasts.append(pred)
        curr_feat = np.roll(curr_feat, 1)
        curr_feat[0, 0] = pred_scaled

    return forecasts

def lstm_forecast(train_scaled, last_sequence, scaler_y, time_steps, days):
    X, y = [], []
    for i in range(time_steps, len(train_scaled)):
        X.append(train_scaled[i-time_steps:i])
        y.append(train_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, batch_size=32, epochs=20, verbose=0)

    forecasts = []
    curr_batch = last_sequence.copy().reshape(1, time_steps, train_scaled.shape[1])

    for _ in range(days):
        pred = model.predict(curr_batch, verbose=0)[0, 0]
        forecasts.append(scaler_y.inverse_transform([[pred]])[0][0])
        new_step = curr_batch[0, -1, :].copy()
        new_step[0] = pred
        new_step = new_step.reshape(1, 1, train_scaled.shape[1])
        curr_batch = np.append(curr_batch[:, 1:, :], new_step, axis=1)

    return forecasts

def calculate_errors(actual, predictions):
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions) * 100
    return rmse, mae, mape

def plot_results(ticker, train_df, test_df, forecasts_df):
    plt.figure(figsize=(14, 7))

    plt.plot(train_df.index[-40:], train_df['Close'].iloc[-40:], label='Historical Train Data', color='gray', linewidth=2)
    plt.plot(test_df.index, test_df['Close'], label='Actual Test Data', color='black', linewidth=3, marker='o')

    colors = ['#e63946', '#2a9d8f', '#e9c46a', '#457b9d']
    for (model_name, preds), color in zip(forecasts_df.items(), colors):
        plt.plot(test_df.index, preds, label=f'{model_name} Forecast', linestyle='dashed', color=color, linewidth=2)

    plt.title(f'{ticker} - 14-Day Model Forecast vs Actual Data', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (INR)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_analysis(ticker, forecast_days=14):
    df = fetch_data(ticker)
    data = engineer_features(df)

    train = data.iloc[:-forecast_days]
    test = data.iloc[-forecast_days:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train.drop(columns=['Close']))
    y_train = scaler_y.fit_transform(train[['Close']])
    last_features = X_train[-1]

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'SVR': SVR(kernel='rbf', C=1000, gamma=0.1)
    }

    results = {}
    for name, model in models.items():
        results[name] = recursive_forecast(model, X_train, y_train, last_features, scaler_y, forecast_days)

    full_scaler = MinMaxScaler()
    train_scaled = full_scaler.fit_transform(train)
    last_sequence = train_scaled[-30:]
    target_scaler = MinMaxScaler()
    target_scaler.fit(train[['Close']])

    results['LSTM'] = lstm_forecast(train_scaled, last_sequence, target_scaler, time_steps=30, days=forecast_days)

    forecasts_df = pd.DataFrame(results, index=test.index)
    
    output_df = forecasts_df.copy()
    output_df.insert(0, 'Actual', test['Close'].values)
    output_df.reset_index(inplace=True)
    output_df.rename(columns={'Date': 'Date'}, inplace=True)
    
    filename = f"{ticker.split('.')[0].lower()}_ml_results.csv"
    output_df.to_csv(filename, index=False)
    print(f"\n>> Successfully saved 14-day predictions to: {filename}")
    # ---------------------------------------------------------

    plot_results(ticker, train, test, forecasts_df)

    print("-" * 50)
    print(f"ERROR ANALYSIS FOR {ticker}")
    print("-" * 50)

    error_metrics = {}
    actual_values = test['Close'].values

    for model_name in results.keys():
        rmse, mae, mape = calculate_errors(actual_values, results[model_name])
        error_metrics[model_name] = {'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape}
        print(f"{model_name}:")
        print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")

    best_model = min(error_metrics, key=lambda k: error_metrics[k]['RMSE'])
    print(f"\n>> BEST PERFORMING MODEL: {best_model} (Lowest RMSE: {error_metrics[best_model]['RMSE']:.4f})")
    print("-" * 50, "\n")

    return forecasts_df

if __name__ == "__main__":
    for ticker in ['RELIANCE.NS', 'IDEA.NS']:
        run_analysis(ticker)