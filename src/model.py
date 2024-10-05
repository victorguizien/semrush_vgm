import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

from src.plot import product_order


def calculate_metrics(actual, predicted):
    """
    Calculate Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

    Parameters:
    actual (array-like): Actual values
    predicted (array-like): Predicted values

    Returns:
    tuple: (MAE, RMSE)
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse


def naive_forecast(data):
    """
    Generate a naive forecast by using the last observed value.

    Parameters:
    data (pd.Series): Time series data

    Returns:
    pd.Series: Naive forecast
    """
    return pd.Series([data.iloc[-1]] * len(data), index=data.index)


def simple_average_forecast(data):
    """
    Generate a forecast using the mean of all historical values.

    Parameters:
    data (pd.Series): Time series data

    Returns:
    pd.Series: Simple average forecast
    """
    return pd.Series([data.mean()] * len(data), index=data.index)


def seasonal_naive_forecast(data, season_length=12):
    """
    Generate a seasonal naive forecast.

    Parameters:
    data (pd.Series): Time series data
    season_length (int): Length of the seasonal cycle

    Returns:
    pd.Series: Seasonal naive forecast
    """
    result = data.copy()
    for i in range(season_length, len(data)):
        result.iloc[i] = data.iloc[i - season_length]
    return result


def simple_moving_average(data, window=3):
    """
    Calculate simple moving average.

    Parameters:
    data (pd.Series): Time series data
    window (int): Size of the moving window

    Returns:
    pd.Series: Simple moving average
    """
    return data.rolling(window=window, min_periods=1).mean()


def naive_forecast_with_ci(data, forecast_horizon=3, confidence=0.95):
    """
    Generate naive forecast with confidence intervals based on historical volatility.

    Parameters:
    data (pd.Series): Time series data
    forecast_horizon (int): Number of periods to forecast
    confidence (float): Confidence level for intervals (0 to 1)

    Returns:
    tuple: (forecasts, lower_ci, upper_ci)
        forecasts (list): Point forecasts
        lower_ci (list): Lower bounds of confidence intervals
        upper_ci (list): Upper bounds of confidence intervals
    """
    last_value = data.iloc[-1]
    forecasts = [last_value] * forecast_horizon

    # Calculate historical volatility (standard deviation of changes)
    changes = np.diff(data)
    volatility = np.std(changes)

    # Calculate confidence interval based on volatility
    margin = stats.norm.ppf((1 + confidence) / 2) * volatility * np.sqrt(np.arange(1, forecast_horizon + 1))
    lower_ci = [last_value - m for m in margin]
    upper_ci = [last_value + m for m in margin]

    return forecasts, lower_ci, upper_ci


def simple_average_forecast_with_ci(data, forecast_horizon=3, confidence=0.95):
    """
    Generate simple average forecast with confidence intervals.

    Parameters:
    data (pd.Series): Time series data
    forecast_horizon (int): Number of periods to forecast
    confidence (float): Confidence level for intervals (0 to 1)

    Returns:
    tuple: (forecasts, lower_ci, upper_ci)
        forecasts (list): Point forecasts (mean of historical data)
        lower_ci (list): Lower bounds of confidence intervals
        upper_ci (list): Upper bounds of confidence intervals
    """
    mean = np.mean(data)
    forecasts = [mean] * forecast_horizon

    # Calculate standard error of the mean
    std_error = np.std(data, ddof=1) / np.sqrt(len(data))

    # Calculate confidence interval
    margin = stats.t.ppf((1 + confidence) / 2, len(data) - 1) * std_error
    lower_ci = [mean - margin] * forecast_horizon
    upper_ci = [mean + margin] * forecast_horizon

    return forecasts, lower_ci, upper_ci


def seasonal_naive_forecast_with_ci(data, forecast_horizon=3, confidence=0.95):
    """
    Generate seasonal naive forecast with confidence intervals.

    Parameters:
    data (pd.Series): Time series data (assumed to be monthly)
    forecast_horizon (int): Number of periods to forecast
    confidence (float): Confidence level for intervals (0 to 1)

    Returns:
    tuple: (forecasts, lower_ci, upper_ci)
        forecasts (list): Seasonal naive point forecasts
        lower_ci (list): Lower bounds of confidence intervals
        upper_ci (list): Upper bounds of confidence intervals
    """
    # Assuming monthly data
    season_length = 12
    forecasts = data[-season_length:-season_length + forecast_horizon]

    # Calculate variability for each month
    monthly_data = [data[i::season_length] for i in range(season_length)]
    monthly_std = [np.std(month, ddof=1) for month in monthly_data]

    # Calculate confidence interval
    margin = [stats.t.ppf((1 + confidence) / 2, len(month) - 1) * std
              for month, std in zip(monthly_data, monthly_std)]
    lower_ci = [f - m for f, m in zip(forecasts, margin[-forecast_horizon:])]
    upper_ci = [f + m for f, m in zip(forecasts, margin[-forecast_horizon:])]

    return forecasts, lower_ci, upper_ci


def simple_moving_average_forecast_with_ci(data, forecast_horizon=3, window_size=3, confidence=0.95):
    """
    Generate simple moving average forecast with confidence intervals.

    Parameters:
    data (pd.Series): Time series data
    forecast_horizon (int): Number of periods to forecast
    window_size (int): Size of the moving average window
    confidence (float): Confidence level for intervals (0 to 1)

    Returns:
    tuple: (forecasts, lower_ci, upper_ci)
        forecasts (list): Simple moving average point forecasts
        lower_ci (list): Lower bounds of confidence intervals
        upper_ci (list): Upper bounds of confidence intervals
    """
    forecasts = []
    lower_ci = []
    upper_ci = []
    working_data = list(data)

    for _ in range(forecast_horizon):
        window = working_data[-window_size:]
        forecast = np.mean(window)
        forecasts.append(forecast)

        # Calculate standard error of forecast
        std_error = np.std(window, ddof=1) / np.sqrt(window_size)

        # Calculate confidence interval
        margin = stats.t.ppf((1 + confidence) / 2, window_size - 1) * std_error
        lower_ci.append(forecast - margin)
        upper_ci.append(forecast + margin)

        working_data.append(forecast)

    return forecasts, lower_ci, upper_ci


def get_baseline_predictions(mrr_monthly):
    """
    Calculate baseline predictions and metrics for multiple forecasting methods.

    This function computes predictions, performance metrics, and future forecasts
    with confidence intervals for four baseline methods: Naive, Simple Average,
    Seasonal Naive, and Simple Moving Average.

    Parameters:
    mrr_monthly (dict): Monthly MRR data for each product

    Returns:
    tuple: (baseline_metrics, baseline_predictions, baseline_future_predictions)
        baseline_metrics (dict): MAE and RMSE for each method and product
        baseline_predictions (dict): Historical predictions for each method and product
        baseline_future_predictions (dict): Future predictions with confidence intervals
                                            for each method and product

    Note:
    The function uses a global 'product_order' list to determine which products to process.
    """
    print("Calculating baseline predictions..")

    baseline_metrics = {}
    baseline_predictions = {}
    baseline_future_predictions = {}

    for product in product_order:
        data = mrr_monthly[product]

        # Naive Forecast
        naive_pred = naive_forecast(data)
        naive_mae, naive_rmse = calculate_metrics(data[1:], naive_pred[1:])

        # Simple Average
        avg_pred = simple_average_forecast(data)
        avg_mae, avg_rmse = calculate_metrics(data, avg_pred)

        # Seasonal Naive
        seasonal_pred = seasonal_naive_forecast(data)
        seasonal_mae, seasonal_rmse = calculate_metrics(data[12:], seasonal_pred[12:])

        # Simple Moving Average
        sma_pred = simple_moving_average(data)
        sma_mae, sma_rmse = calculate_metrics(data[3:], sma_pred[3:])

        baseline_metrics[product] = {
            'Naive': {'MAE': naive_mae, 'RMSE': naive_rmse},
            'Simple Average': {'MAE': avg_mae, 'RMSE': avg_rmse},
            'Seasonal Naive': {'MAE': seasonal_mae, 'RMSE': seasonal_rmse},
            'Simple Moving Average': {'MAE': sma_mae, 'RMSE': sma_rmse}
        }

        baseline_predictions[product] = {
            'Actual': data,
            'Naive': naive_pred,
            'Simple Average': avg_pred,
            'Seasonal Naive': seasonal_pred,
            'Simple Moving Average': sma_pred
        }

        naive_future, naive_lower, naive_upper = naive_forecast_with_ci(data)
        avg_future, avg_lower, avg_upper = simple_average_forecast_with_ci(data)
        seasonal_future, seasonal_lower, seasonal_upper = seasonal_naive_forecast_with_ci(data)
        sma_future, sma_lower, sma_upper = simple_moving_average_forecast_with_ci(data)

        baseline_future_predictions[product] = {
            'Naive': {'forecast': naive_future, 'lower_ci': naive_lower, 'upper_ci': naive_upper},
            'Simple Average': {'forecast': avg_future, 'lower_ci': avg_lower, 'upper_ci': avg_upper},
            'Seasonal Naive': {'forecast': seasonal_future, 'lower_ci': seasonal_lower, 'upper_ci': seasonal_upper},
            'Simple Moving Average': {'forecast': sma_future, 'lower_ci': sma_lower, 'upper_ci': sma_upper}
        }

    print("Done calculating baseline predictions.")

    return baseline_metrics, baseline_predictions, baseline_future_predictions


def display_baseline_metrics_and_predictions(baseline_metrics, baseline_future_predictions):
    """
    Display baseline metrics and future predictions for each product.

    This function prints the MAE and RMSE for each baseline model, as well as
    future predictions with 95% confidence intervals for each product.

    Parameters:
    baseline_metrics (dict): MAE and RMSE for each method and product
    baseline_future_predictions (dict): Future predictions with confidence intervals
                                        for each method and product

    Returns:
    None: This function prints output to the console

    Note:
    The function uses a global 'product_order' list to determine which products to display.
    """
    for product in product_order:
        print(f"\nMetrics and Predictions for {product}:")

        # Baseline
        print("Baseline Metrics:")
        for model, scores in baseline_metrics[product].items():
            print(f"  {model}:")
            print(f"    MAE: {scores['MAE']:.2f}")
            print(f"    RMSE: {scores['RMSE']:.2f}")

        # Future preds
        print("\nFuture Predictions (with 95% Confidence Intervals):")
        for model, predictions in baseline_future_predictions[product].items():
            print(f"  {model}:")
            for i, (forecast, lower, upper) in enumerate(zip(predictions['forecast'],
                                                            predictions['lower_ci'],
                                                            predictions['upper_ci']), 1):
                print(f"    Month {i}: {forecast:.2f} (95% CI: {lower:.2f} - {upper:.2f})")

        print("\n" + "=" * 50)


# ------------------------------------------------------------------------------------------------------------------------
# ARIMA

def auto_arima_optimization(data):
    """
    Perform automatic ARIMA model selection using auto_arima.

    This function uses the auto_arima algorithm to find the best ARIMA
    model parameters for the given time series data.

    Parameters:
    data (pd.Series): Time series data

    Returns:
    tuple: (order, seasonal_order)
        order (tuple): Non-seasonal ARIMA order (p,d,q)
        seasonal_order (tuple): Seasonal ARIMA order (P,D,Q,m)
    """
    model = auto_arima(data, seasonal=True, m=12, start_p=0, start_q=0, max_p=2, max_q=2,
                       start_P=0, start_Q=0, max_P=1, max_Q=1, D=1, max_D=1,
                       trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    return model.order, model.seasonal_order


def analyze_seasonality(data):
    """
    Perform seasonal decomposition and calculate seasonal strength.

    This function decomposes the time series into trend, seasonal, and
    residual components, and calculates the strength of seasonality.

    Parameters:
    data (pd.Series): Time series data

    Returns:
    tuple: (decomposition, seasonal_strength)
        decomposition (DecomposeResult): Seasonal decomposition result
        seasonal_strength (float): Measure of seasonality strength (0 to 1)
    """
    decomposition = seasonal_decompose(data, model='additive', period=12)
    seasonal_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.observed - decomposition.trend)
    return decomposition, seasonal_strength


def forecast_with_ci(model, steps=3, alpha=0.05):
    """
    Generate forecasts with confidence intervals for a fitted SARIMA model.

    Parameters:
    model (SARIMAXResultsWrapper): Fitted SARIMA model
    steps (int): Number of steps to forecast
    alpha (float): Significance level for confidence intervals (default: 0.05 for 95% CI)

    Returns:
    dict: 
        'forecast': List of point forecasts
        'lower_ci': List of lower bounds of confidence intervals
        'upper_ci': List of upper bounds of confidence intervals
    """
    forecast = model.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean.tolist()
    confidence_int = forecast.conf_int(alpha=alpha)

    lower_ci = confidence_int.iloc[:, 0].tolist()
    upper_ci = confidence_int.iloc[:, 1].tolist()

    return {
        'forecast': mean_forecast,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }


def create_sarima_model(data, order, seasonal_order):
    """
    Create and fit a SARIMA model with specified parameters.

    Parameters:
    data (pd.Series): Time series data
    order (tuple): Non-seasonal ARIMA order (p,d,q)
    seasonal_order (tuple): Seasonal ARIMA order (P,D,Q,m)

    Returns:
    SARIMAXResultsWrapper: Fitted SARIMA model
    """
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    return results


def get_sarima_predictions(mrr_monthly):
    """
    Perform SARIMA modeling and forecasting for multiple products.

    This function performs the following steps for each product:
    1. Optimizes SARIMA parameters using auto_arima
    2. Analyzes seasonality
    3. Fits a SARIMA model
    4. Calculates performance metrics (MAE, RMSE, AIC)
    5. Generates forecasts with confidence intervals

    Parameters:
    mrr_monthly (pd.DataFrame): Monthly MRR data for multiple products

    Returns:
    tuple: (sarima_metrics, sarima_forecasts, sarima_observed)
        sarima_metrics (dict): Performance metrics for each product
        sarima_forecasts (dict): Forecasts with confidence intervals for each product
        sarima_observed (dict): Original observed data for each product

    Note:
    The function prints progress and results to the console during execution.
    It handles exceptions that may occur during model fitting.
    """
    sarima_metrics = {}
    sarima_forecasts = {}
    sarima_observed = {}

    for product in mrr_monthly.columns[:]:
        print(f"Optimizing for {product}")

        # Auto ARIMA
        auto_params = auto_arima_optimization(mrr_monthly[product])
        print(f"Best parameters (auto ARIMA): {auto_params}")

        # Analyze seasonality
        decomp, strength = analyze_seasonality(mrr_monthly[product])
        print(f"Seasonal strength: {strength}")

        # Fit
        try:
            model = create_sarima_model(mrr_monthly[product], order=auto_params[0], seasonal_order=auto_params[1])
            print(model.summary())

            # Calculate metrics
            actual = mrr_monthly[product]
            predicted = model.fittedvalues

            # Skip first value due to differencing
            mae, rmse = calculate_metrics(actual[1:], predicted[1:])
            aic = model.aic
            sarima_metrics[product] = {'MAE': mae, 'RMSE': rmse, 'AIC': aic}

            # Forecast with confidence intervals
            dict_forecast = forecast_with_ci(model, steps=3)
            sarima_forecasts[product] = dict_forecast

            # Store original
            sarima_observed[product] = mrr_monthly[product]

        except Exception as e:
            print(f"Error fitting model: {e}")

        print("\n" + "=" * 50 + "\n")

    return sarima_metrics, sarima_forecasts, sarima_observed


def display_sarima_metrics_and_predictions(sarima_metrics, sarima_forecasts, mrr_monthly):
    """
    Display SARIMA model metrics and forecasts for multiple products.

    This function prints:
    1. SARIMA metrics (MAE, RMSE, AIC) for each product
    2. Three-month forecasts with 95% confidence intervals for each product

    Parameters:
    sarima_metrics (dict): Performance metrics for each product
    sarima_forecasts (dict): Forecasts with confidence intervals for each product
    mrr_monthly (pd.DataFrame): Original monthly MRR data used for date referencing

    Returns:
    None: This function prints output to the console

    Note:
    Forecast dates are generated based on the last date in mrr_monthly.
    """
    # Print SARIMA metrics
    for product, metrics in sarima_metrics.items():
        print(f"\nSARIMA metrics for {product}:")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  AIC: {metrics['AIC']:.2f}")

    # Print forecasts with confidence intervals
    for product, forecast_data in sarima_forecasts.items():
        print(f"\n{product} MRR Forecast:")
        forecast_index = pd.date_range(start=mrr_monthly.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')

        for date, mean, lower, upper in zip(forecast_index,
                                            forecast_data['forecast'],
                                            forecast_data['lower_ci'],
                                            forecast_data['upper_ci']):
            print(f"{date.strftime('%Y-%m')}: ${mean:.2f} (95% CI: ${lower:.2f} - ${upper:.2f})")


# ------------------------------------------------------------------------------------------------------------------------
# PROPHET

def calculate_prophet_metrics(model, actual_data):
    """
    Calculate MAE and RMSE for a fitted Prophet model.

    Parameters:
    model (Prophet): Fitted Prophet model
    actual_data (pd.DataFrame): DataFrame with 'ds' and 'y' columns

    Returns:
    tuple: (MAE, RMSE)
    """
    forecast = model.predict(actual_data)
    mae = mean_absolute_error(actual_data['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(actual_data['y'], forecast['yhat']))
    return mae, rmse


def get_prophet_predictions(mrr_monthly):
    """
    Fit Prophet models and generate predictions for multiple products.

    Parameters:
    mrr_monthly (pd.DataFrame): Monthly MRR data for multiple products

    Returns:
    tuple: (prophet_data, prophet_metrics, prophet_forecasts)
        prophet_data (dict): Reformatted data for Prophet for each product
        prophet_metrics (dict): MAE and RMSE for each product's model
        prophet_forecasts (dict): Prophet forecasts for each product
    """
    prophet_data = {}

    for product in mrr_monthly.columns[1:]:
        prophet_data[product] = mrr_monthly[['mrr_month', product]].rename(columns={'mrr_month': 'ds', product: 'y'})

    prophet_forecasts = {}
    prophet_metrics = {}
    for product, data in prophet_data.items():
        model = Prophet()
        model.fit(data)

        mae, rmse = calculate_prophet_metrics(model, data)
        prophet_metrics[product] = {'MAE': mae, 'RMSE': rmse}

        future = model.make_future_dataframe(periods=3, freq='MS')

        forecast = model.predict(future)
        prophet_forecasts[product] = forecast

    return prophet_data, prophet_metrics, prophet_forecasts


def display_metrics_and_predictions(prophet_metrics, prophet_forecasts):
    """
    Display Prophet model metrics and forecasts for multiple products.

    This function prints:
    1. MAE and RMSE for each product's Prophet model
    2. Three-month forecasts with 95% confidence intervals for each product

    Parameters:
    prophet_metrics (dict): MAE and RMSE for each product's model
    prophet_forecasts (dict): Prophet forecasts for each product

    Returns:
    None: This function prints output to the console
    """
    for product, metrics in prophet_metrics.items():
        print(f"\nMetrics for {product}:")
        print(f"MAE: {metrics['MAE']}")
        print(f"RMSE: {metrics['RMSE']}")

    for product, forecast in prophet_forecasts.items():
        print(f"\n{product} MRR Forecast:")
        future_forecast = forecast.iloc[-3:]
        for _, row in future_forecast.iterrows():
            print(f"{row['ds'].strftime('%Y-%m')}: ${row['yhat']:.2f} (95% CI: ${row['yhat_lower']:.2f} - ${row['yhat_upper']:.2f})")


# ------------------------------------------------------------------------------------------------------------------------
def get_comparative_metrics(baseline_metrics, sarima_metrics, prophet_metrics):
    """
    Compile comparative metrics for all models across all products.

    Parameters:
    baseline_metrics (dict): Metrics for baseline models
    sarima_metrics (dict): Metrics for SARIMA models
    prophet_metrics (dict): Metrics for Prophet models

    Returns:
    pd.DataFrame: DataFrame with multi-index (Product, Model) containing MAE, RMSE, and AIC (where applicable)

    Note:
    Uses a global 'product_order' list to determine which products to include.
    """
    data = []

    # Populate the data list
    for product in product_order:
        # Prophet
        data.append({
            'Product': product,
            'Model': 'Prophet',
            'MAE': prophet_metrics[product]['MAE'],
            'RMSE': prophet_metrics[product]['RMSE'],
            'AIC': None
        })

        # SARIMA
        data.append({
            'Product': product,
            'Model': 'SARIMA',
            'MAE': sarima_metrics[product]['MAE'],
            'RMSE': sarima_metrics[product]['RMSE'],
            'AIC': sarima_metrics[product]['AIC']
        })

        # Baseline models
        for baseline in ['Naive', 'Simple Average', 'Seasonal Naive', 'Simple Moving Average']:
            data.append({
                'Product': product,
                'Model': baseline,
                'MAE': baseline_metrics[product][baseline]['MAE'],
                'RMSE': baseline_metrics[product][baseline]['RMSE'],
                'AIC': None
            })
    df_metrics = pd.DataFrame(data)

    # Multi index for nice table
    df_metrics.set_index(['Product', 'Model'], inplace=True)

    return df_metrics


def get_comparative_forecasts(baseline_future_predictions, prophet_forecasts, sarima_forecasts):
    """
    Compile comparative forecasts for all models across all products.

    Parameters:
    baseline_future_predictions (dict): Future predictions for baseline models
    prophet_forecasts (dict): Forecasts from Prophet models
    sarima_forecasts (dict): Forecasts from SARIMA models

    Returns:
    pd.DataFrame: DataFrame with multi-index (Product, Model, Month) containing forecasts and confidence intervals

    Note:
    Uses a global 'product_order' list to determine which products to include.
    Includes 3-month forecasts for each model.
    """
    data = []

    for product in product_order:
        # Baseline models
        for model in ['Naive', 'Simple Average', 'Seasonal Naive', 'Simple Moving Average']:
            for i, (forecast, lower, upper) in enumerate(zip(
                baseline_future_predictions[product][model]['forecast'],
                baseline_future_predictions[product][model]['lower_ci'],
                baseline_future_predictions[product][model]['upper_ci']
            ), 1):
                data.append({
                    'Product': product,
                    'Model': model,
                    'Month': i,
                    'Forecast': forecast,
                    'Lower CI': lower,
                    'Upper CI': upper
                })

        # Prophet
        for i, (forecast, lower, upper) in enumerate(zip(
            prophet_forecasts[product]['yhat'].tail(3),
            prophet_forecasts[product]['yhat_lower'].tail(3),
            prophet_forecasts[product]['yhat_upper'].tail(3),
        ), 1):
            data.append({
                'Product': product,
                'Model': 'Prophet',
                'Month': i,
                'Forecast': forecast,
                'Lower CI': lower,
                'Upper CI': upper
            })

        # SARIMA
        for i, (forecast, lower, upper) in enumerate(zip(
            sarima_forecasts[product]['forecast'],
            sarima_forecasts[product]['lower_ci'],
            sarima_forecasts[product]['upper_ci']
        ), 1):
            data.append({
                'Product': product,
                'Model': 'SARIMA',
                'Month': i,
                'Forecast': forecast,
                'Lower CI': lower,
                'Upper CI': upper
            })

    # Create the DataFrame
    df_forecast_metrics = pd.DataFrame(data)

    # Set multi-index for easy viewing
    df_forecast_metrics.set_index(['Product', 'Model', 'Month'], inplace=True)

    return df_forecast_metrics
