# Install and load required packages
install.packages(c("quantmod", "forecast", "tseries"))

library(quantmod)
library(forecast)
library(tseries)


# Fetch Vodafone Idea stock data
getSymbols("IDEA.NS", src = "yahoo", from = "2020-01-01")

# Use adjusted closing prices
price_data <- Ad(IDEA.NS)
price_vector <- as.numeric(price_data)


# Split data into training and testing (last 14 days for testing)
n <- length(price_vector)

train_data <- price_vector[1:(n - 14)]
test_data  <- price_vector[(n - 13):n]

train_ts <- ts(train_data)


# Check stationarity
cat("ADF Test (Original Series):\n")
print(adf.test(train_ts))


# Fit AR(2) model
cat("\nAR(2) Model:\n")

ar_model <- arima(train_ts, order = c(2, 1, 0))
summary(ar_model)

cat("AIC:", AIC(ar_model), "\n")
cat("BIC:", BIC(ar_model), "\n")

# Residual diagnostics
checkresiduals(ar_model)
Box.test(residuals(ar_model), lag = 20, type = "Ljung-Box")


# Fit MA(2) model
cat("\nMA(2) Model:\n")

ma_model <- arima(train_ts, order = c(0, 1, 2))
summary(ma_model)

cat("AIC:", AIC(ma_model), "\n")
cat("BIC:", BIC(ma_model), "\n")

# Residual diagnostics
checkresiduals(ma_model)
Box.test(residuals(ma_model), lag = 20, type = "Ljung-Box")


# Forecast next 14 days
ar_forecast <- forecast(ar_model, h = 14)
ma_forecast <- forecast(ma_model, h = 14)

ar_pred <- as.numeric(ar_forecast$mean)
ma_pred <- as.numeric(ma_forecast$mean)

actual_values <- test_data


# Calculate accuracy metrics
mse_ar  <- mean((actual_values - ar_pred)^2)
mae_ar  <- mean(abs(actual_values - ar_pred))
mape_ar <- mean(abs((actual_values - ar_pred) / actual_values)) * 100

mse_ma  <- mean((actual_values - ma_pred)^2)
mae_ma  <- mean(abs(actual_values - ma_pred))
mape_ma <- mean(abs((actual_values - ma_pred) / actual_values)) * 100

cat("\nModel Performance:\n")

cat("\nAR(2):\n")
cat("MSE:", mse_ar, "\n")
cat("MAE:", mae_ar, "\n")
cat("MAPE:", mape_ar, "%\n")

cat("\nMA(2):\n")
cat("MSE:", mse_ma, "\n")
cat("MAE:", mae_ma, "\n")
cat("MAPE:", mape_ma, "%\n")


# Plot actual vs predicted values
time_index <- 1:14

plot(time_index, actual_values, type = "l", col = "blue", lwd = 2,
     main = "Actual vs AR(2) vs MA(2) (Vodafone Idea)",
     xlab = "Days", ylab = "Price")

lines(time_index, ar_pred, col = "red", lwd = 2)
lines(time_index, ma_pred, col = "green", lwd = 2)

legend("topleft",
       legend = c("Actual", "AR(2)", "MA(2)"),
       col = c("blue", "red", "green"),
       lty = 1, lwd = 2)


# Save results
results_df <- data.frame(
  Day = 1:14,
  Actual = actual_values,
  AR_Predicted = ar_pred,
  MA_Predicted = ma_pred
)

write.csv(results_df, "idea_ar_ma_results.csv", row.names = FALSE)