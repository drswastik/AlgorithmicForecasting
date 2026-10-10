# Stock Price Forecasting: Econometrics vs. Machine Learning

## Overview
This project explores and compares the efficacy of classical econometric models and modern machine learning algorithms in forecasting daily stock prices. To test the models under different market conditions, I analyzed two distinct stocks:
*   **Reliance Industries (RELIANCE.NS):** A relatively stable and predictable stock.
*   **Vodafone Idea (IDEA.NS):** A highly volatile and unpredictable stock.

## Methodology
I engineered features from historical data and forecasted a 14-day trading horizon using two different analytical approaches:
1.  **Classical Econometrics (R):** Auto-Regressive (AR), Moving Average (MA), and ARIMA models (optimized via AIC/BIC).
2.  **Machine Learning (Python):** Random Forest, XGBoost, Support Vector Regression (SVR), and Long Short-Term Memory (LSTM) neural networks.

## Key Findings
*   **Stable Data (Reliance):** Traditional econometrics outperformed complex AI. The **ARIMA(1,1,0) with drift** model yielded the highest accuracy with a Mean Absolute Percentage Error (MAPE) of just **1.34%**, successfully capturing the stock's upward trend.
*   **Volatile Data (Vodafone Idea):** Classical models failed to capture the volatility, outputting relatively flat forecasts. Deep learning prevailed here, with the **LSTM** model achieving the best accuracy (MAPE: **5.20%**).
*   **Conclusion:** There is no universal "best" model for stock forecasting. Traditional statistical models are highly efficient and accurate for orderly datasets, while deep learning (LSTM) is necessary to navigate noisy, highly volatile time series.

## Technologies Used
*   **Languages:** R, Python
*   **Econometrics (R):** `quantmod`, `forecast`, `tseries`
*   **Machine Learning (Python):** `yfinance`, `pandas`, `scikit-learn`, `xgboost`, `tensorflow` (Keras)

---
