# Load Reliance stock data
getSymbols("RELIANCE.NS", src = "yahoo", from = "2020-01-01")

price_data <- Ad(RELIANCE.NS)
price_vector <- as.numeric(price_data)


# Split into training and test sets (last 14 days as test)
n <- length(price_vector)

train_data <- price_vector[1:(n - 14)]
test_data  <- price_vector[(n - 13):n]

train_ts <- ts(train_data)


# Check stationarity of original series
cat("ADF Test (Original Series):\n")
print(adf.test(train_ts))


# Apply first differencing
diff_ts <- diff(train_ts)

cat("\nADF Test (After Differencing):\n")
print(adf.test(diff_ts))


# Use differenced series for AR and MA model selection
ts_used <- diff_ts


# Try AR models with different orders
ar_results <- data.frame(p = integer(), AIC = numeric(), BIC = numeric())

for (p in 1:5) {
  model <- arima(ts_used, order = c(p, 0, 0))
  
  ar_results <- rbind(ar_results,
                      data.frame(p = p,
                                 AIC = AIC(model),
                                 BIC = BIC(model)))
}

cat("\nAR Model Selection:\n")
print(ar_results)


# Try MA models with different orders
ma_results <- data.frame(q = integer(), AIC = numeric(), BIC = numeric())

for (q in 1:5) {
  model <- arima(ts_used, order = c(0, 0, q))
  
  ma_results <- rbind(ma_results,
                      data.frame(q = q,
                                 AIC = AIC(model),
                                 BIC = BIC(model)))
}

cat("\nMA Model Selection:\n")
print(ma_results)


# Based on AIC, select AR(1) and MA(1)

cat("\nFinal AR(1) Model:\n")
ar_model <- arima(train_ts, order = c(1, 1, 0))
summary(ar_model)

cat("\nFinal MA(1) Model:\n")
ma_model <- arima(train_ts, order = c(0, 1, 1))
summary(ma_model)


# Check residuals for both models
cat("\nAR(1) Residual Check:\n")
checkresiduals(ar_model)
Box.test(residuals(ar_model), lag = 20, type = "Ljung-Box")

cat("\nMA(1) Residual Check:\n")
checkresiduals(ma_model)
Box.test(residuals(ma_model), lag = 20, type = "Ljung-Box")


# Forecast next 14 days
ar_forecast <- forecast(ar_model, h = 14)
ma_forecast <- forecast(ma_model, h = 14)

ar_pred <- as.numeric(ar_forecast$mean)
ma_pred <- as.numeric(ma_forecast$mean)

actual_values <- test_data


# Calculate performance metrics
mse_ar  <- mean((actual_values - ar_pred)^2)
mae_ar  <- mean(abs(actual_values - ar_pred))
mape_ar <- mean(abs((actual_values - ar_pred) / actual_values)) * 100

mse_ma  <- mean((actual_values - ma_pred)^2)
mae_ma  <- mean(abs(actual_values - ma_pred))
mape_ma <- mean(abs((actual_values - ma_pred) / actual_values)) * 100

cat("\nModel Performance:\n")

cat("\nAR(1):\n")
cat("MSE:", mse_ar, "\n")
cat("MAE:", mae_ar, "\n")
cat("MAPE:", mape_ar, "%\n")

cat("\nMA(1):\n")
cat("MSE:", mse_ma, "\n")
cat("MAE:", mae_ma, "\n")
cat("MAPE:", mape_ma, "%\n")


# Plot actual vs predicted values
time_index <- 1:14

plot(time_index, actual_values, type = "l", col = "blue", lwd = 2,
     main = "Actual vs AR(1) vs MA(1) - Reliance",
     xlab = "Days", ylab = "Price")

lines(time_index, ar_pred, col = "red", lwd = 2)
lines(time_index, ma_pred, col = "green", lwd = 2)

legend("topleft",
       legend = c("Actual", "AR(1)", "MA(1)"),
       col = c("blue", "red", "green"),
       lty = 1, lwd = 2)


# Save results for later comparison
results_df <- data.frame(
  Day = 1:14,
  Actual = actual_values,
  AR_Predicted = ar_pred,
  MA_Predicted = ma_pred
)

write.csv(results_df, "reliance_ar_ma_results.csv", row.names = FALSE)