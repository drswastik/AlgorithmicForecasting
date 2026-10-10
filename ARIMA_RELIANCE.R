# Fetch Reliance stock data from Yahoo Finance
getSymbols("RELIANCE.NS", src = "yahoo", from = "2020-01-01")

# Work with adjusted closing prices
price_data <- Ad(RELIANCE.NS)
price_vector <- as.numeric(price_data)


# Split into training and testing (last 14 days kept aside)
n <- length(price_vector)

train_data <- price_vector[1:(n - 14)]
test_data  <- price_vector[(n - 13):n]

train_ts <- ts(train_data)


# Check whether the series is stationary
cat("ADF Test:\n")
print(adf.test(train_ts))


# Try a few ARIMA models and compare using AIC
model1 <- arima(train_ts, order = c(1,1,0))
model2 <- arima(train_ts, order = c(1,1,1))
model3 <- arima(train_ts, order = c(2,1,1))
model4 <- arima(train_ts, order = c(2,1,2))

cat("\nAIC Values:\n")
cat("ARIMA(1,1,0):", AIC(model1), "\n")
cat("ARIMA(1,1,1):", AIC(model2), "\n")
cat("ARIMA(2,1,1):", AIC(model3), "\n")
cat("ARIMA(2,1,2):", AIC(model4), "\n")


# Pick the best model based on AIC (update if needed)
best_model <- model1
summary(best_model)


# Add drift to capture any trend in the data
best_model <- Arima(train_ts, order = c(1,1,0), include.drift = TRUE)
summary(best_model)


# Check residuals to make sure the model is valid
checkresiduals(best_model)
Box.test(residuals(best_model), lag = 20, type = "Ljung-Box")


# Forecast the next 14 days
forecast_values <- forecast(best_model, h = 14)

predicted <- as.numeric(forecast_values$mean)
actual_values <- test_data


# Calculate performance metrics
mse  <- mean((actual_values - predicted)^2)
mae  <- mean(abs(actual_values - predicted))
mape <- mean(abs((actual_values - predicted) / actual_values)) * 100

cat("\nARIMA Performance:\n")
cat("MSE:", mse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")


# Plot actual vs predicted values
time_index <- 1:14

plot(time_index, actual_values, type = "l", col = "blue", lwd = 2,
     main = "Actual vs ARIMA (Reliance)",
     xlab = "Days", ylab = "Price")

lines(time_index, predicted, col = "red", lwd = 2)

legend("topleft",
       legend = c("Actual", "ARIMA"),
       col = c("blue", "red"),
       lty = 1, lwd = 2)


# Save results for reporting
results_df <- data.frame(
  Day = 1:14,
  Actual = actual_values,
  Predicted = predicted
)

write.csv(results_df, "reliance_arima_results.csv", row.names = FALSE)