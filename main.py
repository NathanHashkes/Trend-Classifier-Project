import sys
import config
from data_io import choose_file, load_file, get_ticker_data, get_model_input_data
from model import get_predictions
from anomaly import detect_anomaly
from trend import detect_trend



def main():
    # --- Data ---
    path = choose_file()
    data = load_file(path, config.REQUIRED_COLUMNS)
    ticker_name = input("Enter ticker: ")
    stock_data = get_ticker_data(data, ticker_name) # Also prints how many rows - and what dates - it got.

    # --- Anomaly ---
    scan_date = config.SCAN_DATE # Can change scan date here
    anomaly = detect_anomaly(stock_data, scan_date=scan_date)
 
    if anomaly is None:
        print(f"No anomaly found at {scan_date}.")
        return
 
    print(f"Anomaly detected between {anomaly.start} and {anomaly.end}.")
    print(f"Length: {anomaly.length} days | z = {anomaly.z_score}\n")

    # --- Model ---
    data = load_file(path, config.REQUIRED_MODEL_COLUMNS)
    model_input = get_model_input_data(data, ticker_name, scan_date)
    horizon = max(config.TREND_LENGTH)
    predicted_returns = get_predictions(model_input, horizon)

    # --- Trend ---
    trend_label = detect_trend(anomaly, predicted_returns)

    if trend_label == 1:
        print(f"The anomaly between {anomaly.start} and {anomaly.end} will develop into a trend in the next {horizon} days")
    else:
        print(f"The anomaly between {anomaly.start} and {anomaly.end} will NOT develop into a trend in the next {horizon} days")



if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        sys.exit(f"\nError: {exc}\n")