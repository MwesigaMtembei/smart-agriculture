from prophet import Prophet
import pandas as pd

def forecast_crop_prices(csv_file, periods=2):
    # Load and prepare data
    df = pd.read_csv(csv_file)
    df.rename(columns={"date": "ds", "price": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])

    # ✅ Drop rows where price is missing (NaN)
    df = df.dropna(subset=["y"])

    # ✅ Place this line right after loading and renaming the columns
    if len(df) < 2:
        return "not_enough_data"

    # Fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Predict future dates
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)

    # Return only relevant forecast columns
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
