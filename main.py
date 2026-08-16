import sys
import config
from data_io import choose_file, load_file, get_ticker_data
from model import get_predictions
from anomaly import detect_anomaly
from trend import detect_trend

def main():
    # --- Data ---
    path = choose_file()
    data = load_file(path)
    ticker_name = input("Enter ticker: ")
    stock_data = get_ticker_data(data, ticker_name) # Also prints how many rows - and what dates - it got.

    # --- Anomaly ---
    scan_date = config.SCAN_DATE
    anomaly = detect_anomaly(stock_data, scan_date=scan_date) # Can change scan date here
 
    if anomaly is None:
        print(f"No anomaly found at {scan_date}.")
        return
 
    print(f"Anomaly detected between {anomaly.start} and {anomaly.end}.")
    print(f"Length: {anomaly.length} days | z = {anomaly.z_score}\n")

    # --- Model ---
    # TODO put model code here
    horizon = max(config.TREND_LENGTH)
    predicted_returns = get_predictions(horizon)

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

# This is a test commit 1